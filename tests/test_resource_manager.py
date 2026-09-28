"""
Test suite for Autonomous System Resource Manager.
Verifies hardware profiling, adaptive memory management, SQLite WAL optimization,
CPU thread budgeting, token budgeting, and system integration.
"""

import gc
import os
import sqlite3
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.system.resource_manager import (
    ResourceTier,
    SystemResourceProfile,
    AdaptiveMemoryManager,
    SQLiteOptimizer,
    CPUBudgeter,
    TokenBudgeter,
    ResourceManager,
    get_resource_manager,
)
from src.memory.session import SessionManager
from src.memory.episodic import EpisodicMemoryManager
from src.retrieval.wikipedia_client import WikipediaCache
from src.reasoning.xthinking import XThinkingEngine


class TestSystemResourceProfile(unittest.TestCase):
    """Tests hardware capability detection and tier classification."""

    def test_profile_detection(self):
        profile = SystemResourceProfile.detect()
        self.assertGreater(profile.total_ram_gb, 0.0)
        self.assertGreater(profile.available_ram_gb, 0.0)
        self.assertGreaterEqual(profile.ram_usage_pct, 0.0)
        self.assertLessEqual(profile.ram_usage_pct, 100.0)
        self.assertGreaterEqual(profile.cpu_count_physical, 1)
        self.assertGreaterEqual(profile.cpu_count_logical, 1)
        self.assertGreater(profile.process_rss_mb, 0.0)
        self.assertIn(profile.tier, [ResourceTier.CONSTRAINED, ResourceTier.BALANCED, ResourceTier.PERFORMANCE])

    def test_profile_serialization(self):
        profile = SystemResourceProfile.detect()
        d = profile.to_dict()
        self.assertIn("tier", d)
        self.assertIn("is_low_end", d)
        self.assertIn("total_ram_gb", d)
        self.assertIn("cpu_count_physical", d)

        summary = profile.formatted_summary()
        self.assertIn(profile.tier.value, summary)
        self.assertIn("RAM", summary)

    def test_tier_classification_logic(self):
        # Test simulated constrained profile
        low_profile = SystemResourceProfile(
            total_ram_gb=3.8,
            available_ram_gb=0.8,
            used_ram_gb=3.0,
            ram_usage_pct=78.9,
            cpu_count_physical=2,
            cpu_count_logical=2,
            cpu_percent=10.0,
            process_rss_mb=45.0,
            has_gpu=False,
            gpu_name="None",
            tier=ResourceTier.CONSTRAINED,
            disk_free_gb=10.0,
        )
        self.assertTrue(low_profile.is_low_end)
        self.assertEqual(low_profile.tier, ResourceTier.CONSTRAINED)


class TestAdaptiveMemoryManager(unittest.TestCase):
    """Tests memory tracking, proactive garbage collection, and working set reclamation."""

    def setUp(self):
        self.mem_mgr = AdaptiveMemoryManager()

    def test_process_rss_reading(self):
        rss = self.mem_mgr.get_process_rss_mb()
        self.assertGreater(rss, 0.0)
        self.assertLess(rss, 16000.0)  # reasonable bounds

    def test_cache_clearer_registration(self):
        cleared = False

        def my_clearer():
            nonlocal cleared
            cleared = True

        self.mem_mgr.register_cache_clearer(my_clearer)
        self.mem_mgr.cleanup_after_turn(turn_index=1, force=True)
        self.assertTrue(cleared)

    def test_allocation_and_reclamation(self):
        initial_rss = self.mem_mgr.get_process_rss_mb()

        # Allocate 30MB temporary buffer
        temp_data = [bytearray(1024 * 1024) for _ in range(30)]
        allocated_rss = self.mem_mgr.get_process_rss_mb()
        self.assertGreater(allocated_rss, initial_rss)

        # Release and trigger cleanup
        del temp_data
        result = self.mem_mgr.cleanup_after_turn(turn_index=1, force=True)

        self.assertTrue(result["performed"])
        self.assertGreaterEqual(result["freed_mb"], 0.0)
        self.assertGreaterEqual(result["gc_objects_collected"], 0)

        # Confirm working set trimmed
        after_rss = self.mem_mgr.get_process_rss_mb()
        self.assertLess(after_rss, allocated_rss)


class TestSQLiteOptimizer(unittest.TestCase):
    """Tests high-speed WAL pragma application across databases."""

    def setUp(self):
        gc.collect()
        self.test_db_path = Path("test_resource_opt.db")
        if self.test_db_path.exists():
            try:
                self.test_db_path.unlink()
            except Exception:
                pass

    def tearDown(self):
        gc.collect()
        if self.test_db_path.exists():
            try:
                self.test_db_path.unlink()
            except Exception:
                pass
        wal_file = Path("test_resource_opt.db-wal")
        shm_file = Path("test_resource_opt.db-shm")
        for f in (wal_file, shm_file):
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

    def test_optimize_connection(self):
        conn = sqlite3.connect(self.test_db_path)
        conn.execute("CREATE TABLE test (id INT, val TEXT);")
        SQLiteOptimizer.optimize_connection(conn, tier=ResourceTier.BALANCED)

        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
        temp_store = conn.execute("PRAGMA temp_store;").fetchone()[0]

        self.assertEqual(journal.lower(), "wal")
        self.assertEqual(sync, 1)  # 1 = NORMAL
        self.assertEqual(temp_store, 2)  # 2 = MEMORY
        conn.close()

    def test_optimize_database_file(self):
        with sqlite3.connect(self.test_db_path) as conn:
            conn.execute("CREATE TABLE test_data (id INT, name TEXT);")

        status = SQLiteOptimizer.optimize_database(self.test_db_path, tier=ResourceTier.CONSTRAINED)
        self.assertEqual(status["status"], "optimized")
        self.assertEqual(status["journal_mode"].lower(), "wal")

    def test_get_optimized_connection_factory(self):
        conn = SQLiteOptimizer.get_optimized_connection(self.test_db_path, tier=ResourceTier.PERFORMANCE)
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        self.assertEqual(journal.lower(), "wal")
        conn.close()


class TestCPUBudgeter(unittest.TestCase):
    """Tests CPU concurrency controls and thread pool limits."""

    def setUp(self):
        self.cpu = CPUBudgeter()

    def test_concurrency_environment_config(self):
        env_vars = self.cpu.configure_concurrency_environment(ResourceTier.CONSTRAINED)
        self.assertIn("OMP_NUM_THREADS", env_vars)
        self.assertIn("MKL_NUM_THREADS", env_vars)
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), env_vars["OMP_NUM_THREADS"])

    def test_worker_counts_by_tier(self):
        io_constrained = self.cpu.get_optimal_worker_count("io", ResourceTier.CONSTRAINED)
        io_perf = self.cpu.get_optimal_worker_count("io", ResourceTier.PERFORMANCE)
        self.assertLessEqual(io_constrained, io_perf)

        cpu_constrained = self.cpu.get_optimal_worker_count("cpu", ResourceTier.CONSTRAINED)
        self.assertEqual(cpu_constrained, 1)  # Strict 1 CPU worker on constrained machines

        bg_constrained = self.cpu.get_optimal_worker_count("background", ResourceTier.CONSTRAINED)
        bg_perf = self.cpu.get_optimal_worker_count("background", ResourceTier.PERFORMANCE)
        self.assertLess(bg_constrained, bg_perf)


class TestTokenBudgeter(unittest.TestCase):
    """Tests dynamic token budgeting and prompt compaction."""

    def setUp(self):
        self.tb = TokenBudgeter()

    def test_base_thinking_budgets(self):
        b_constrained = self.tb.get_thinking_base_budget(ResourceTier.CONSTRAINED)
        b_balanced = self.tb.get_thinking_base_budget(ResourceTier.BALANCED)
        b_perf = self.tb.get_thinking_base_budget(ResourceTier.PERFORMANCE)

        self.assertEqual(b_constrained, 1024)
        self.assertEqual(b_balanced, 4096)
        self.assertEqual(b_perf, 8192)

    def test_adaptive_thinking_budget_scaling(self):
        # D = 0.0 -> base budget
        b0 = self.tb.calculate_thinking_budget(0.0, ResourceTier.CONSTRAINED)
        self.assertEqual(b0, 1024)

        # D = 0.8 -> scaled budget
        b_high = self.tb.calculate_thinking_budget(0.8, ResourceTier.CONSTRAINED)
        self.assertGreater(b_high, 1024)
        self.assertLessEqual(b_high, 4096)

    def test_generation_limits(self):
        lim_constrained = self.tb.get_generation_limits(ResourceTier.CONSTRAINED)
        lim_perf = self.tb.get_generation_limits(ResourceTier.PERFORMANCE)

        self.assertEqual(lim_constrained["context_preset"], "small")
        self.assertEqual(lim_constrained["max_new_tokens"], 1024)
        self.assertEqual(lim_perf["context_preset"], "large")
        self.assertEqual(lim_perf["max_new_tokens"], 4096)

    def test_prompt_optimization(self):
        raw_prompt = "Line 1\n\n\n\n\nLine 2    \n\n\nLine 3"
        compact = self.tb.optimize_prompt(raw_prompt, ResourceTier.CONSTRAINED)
        self.assertNotIn("\n\n\n", compact)
        self.assertIn("Line 1\n\nLine 2\n\nLine 3", compact)


class TestResourceManagerIntegration(unittest.TestCase):
    """Tests system-wide integration of the Resource Manager with engine components."""

    def test_singleton_initialization(self):
        rm = get_resource_manager()
        self.assertIsNotNone(rm)
        status = rm.status_dict()
        self.assertIn("profile", status)
        self.assertIn("memory", status)
        self.assertIn("cpu", status)
        self.assertIn("token_budget", status)

    def test_session_manager_optimized_connection(self):
        test_session_db = Path("test_session_opt.db")
        if test_session_db.exists():
            test_session_db.unlink()

        try:
            sm = SessionManager(db_path=test_session_db)
            s_id = sm.create_session("Test Session")
            self.assertIsNotNone(s_id)

            with sm._get_connection() as conn:
                journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
                sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
                self.assertEqual(journal.lower(), "wal")
                self.assertEqual(sync, 1)
        finally:
            if test_session_db.exists():
                try:
                    test_session_db.unlink()
                except Exception:
                    pass
            for f in (Path("test_session_opt.db-wal"), Path("test_session_opt.db-shm")):
                if f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass

    def test_xthinking_engine_resource_manager_attachment(self):
        engine = XThinkingEngine(research_mode="off")
        self.assertTrue(hasattr(engine, "resource_mgr"))
        self.assertIsNotNone(engine.resource_mgr)
        self.assertIn(engine.resource_mgr.profile.tier, [ResourceTier.CONSTRAINED, ResourceTier.BALANCED, ResourceTier.PERFORMANCE])


if __name__ == "__main__":
    unittest.main()

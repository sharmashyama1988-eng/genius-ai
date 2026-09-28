# 🛠️ Genius AI — Custom Skills & Scripts System (`agent/`)

> **Apni khud ki scripts likho — Genius AI automatically discover karega aur use karega!**  
> **Write your own scripts and tools with ZERO limits. Genius AI automatically discovers, understands, and executes them.**

---

## 🌟 Yeh Kaise Kaam Karta Hai? (How It Works)

Aapko AI ko modify karne ki ya code inject karne ki zaroorat nahi hai.  
Sirf `agent/skills/` folder mein apni nayi script `.py` ya folder add karo:

```
agent/
├── registry.py                # Auto-discovery engine (Genius AI uses this)
├── README.md                  # Yeh documentation
└── skills/                    # Yahan aapki saari skills rahengi
    ├── pdf_generator.py       # Built-in: PDF generation skill
    ├── excel_generator.py     # Built-in: Excel .xlsx generation skill
    ├── ppt_generator.py       # Built-in: PowerPoint .pptx generation skill
    ├── imagen/                # Built-in: Image generation skill (Pollinations / API)
    │   ├── __init__.py
    │   └── image_generator.py
    └── [aapki_nayi_skill].py  # 👈 Yahan apna koi bhi naya script daalo!
```

Jab aap Genius AI se koi kaam bologe, AI:
1. `agent/skills/` ko scan karega.
2. Aapki script ka docstring aur functions padhega.
3. Automatically samajh jaayega ki yeh kaam aapki script se ho sakta hai.
4. Script ko run karega, output capture karega, aur result aapko return karega!

---

## 🚀 Pre-Built Ready-to-Use Skills (Pahale se bane huye tools)

| Skill | Folder / File | Kya karta hai | Usage Example |
|---|---|---|---|
| 📄 **PDF Generator** | `skills/pdf_generator.py` | Professional multi-page PDFs with cover, tables, page numbers, and callouts | `python -m agent.skills.pdf_generator -o doc.pdf -t "Title"` |
| 📊 **Excel Generator** | `skills/excel_generator.py` | Styled `.xlsx` sheets with formulas, totals, currency, and auto-column width | `python -m agent.skills.excel_generator -o data.xlsx` |
| 📽️ **PPT Generator** | `skills/ppt_generator.py` | Modern 16:9 widescreen PowerPoint decks with dark/navy themes and KPI stat cards | `python -m agent.skills.ppt_generator -o deck.pptx -t "AI Strategy"` |
| 🎨 **Imagen (Image)** | `skills/imagen/` | Text-to-Image generator (Free zero-key Pollinations + DALL-E / Custom API) | `python -m agent.skills.imagen.image_generator -p "cyberpunk city" -o img.png` |

---

## 📝 Apna Naya Skill Kaise Banayein? (How to Add Your Own Skill)

### Example 1: Single File Skill (`agent/skills/qr_code.py`)

Aapko ek QR code generator skill add karni hai:

1. `agent/skills/qr_code.py` banao:
```python
"""QR Code Generator Skill for Genius AI.
Generates QR code PNG images for URLs or text.
"""
import sys

def make_qr(text: str, output_path: str = "qr.png"):
    import qrcode
    img = qrcode.make(text)
    img.save(output_path)
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", required=True)
    parser.add_argument("-o", "--output", default="qr.png")
    args = parser.parse_args()
    out = make_qr(args.text, args.output)
    print(f"SUCCESS: Saved QR to {out}")
```

2. **Bas ho gaya!** Ab Genius terminal mein bas bolo:
   `"https://google.com ka QR code bana do"`
   Genius AI automatically `agent/skills/qr_code.py` run karke QR code save karega aur path de dega!

---

### Example 2: Folder-Based Skill (`agent/skills/my_scraper/`)

Agar aapka tool complex hai aur multiple files ya API credentials chahiye:

1. Folder banao: `agent/skills/my_scraper/`
2. `__init__.py` aur `scraper.py` likho.
3. Top par docstring likho jo describe kare ki tool kya karta hai.
4. CLI arguments provide karo (`if __name__ == '__main__': ...`).

Genius AI folder ke naam se skill detect kar lega aur autonomous mode mein direct execute karega!

---

## ⚡ Zero Limits (Koi Bandish Nahi)

Aap yahan kuch bhi add kar sakte ho:
- **Web Scrapers** (BeautifulSoup, Playwright, Scrapy)
- **Audio / TTS / Speech** (Edge-TTS, Whisper, ElevenLabs)
- **Video Editing** (MoviePy, ffmpeg-python)
- **Database Connectors** (PostgreSQL, MongoDB, BigQuery)
- **Custom AI / ML Models** (Local ONNX, Hugging Face pipelines)
- **Financial APIs** (Stock market, Crypto, Yahoo Finance)
- **IoT & Hardware control** (Serial, Arduino, Raspberry Pi)

Genius AI automatically detects the skill, passes inputs, handles errors, and returns clean results!

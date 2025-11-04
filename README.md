# SUI_BOT_Z — Smart SUI/USDT Bot (Bybit Perp)

بوت تداول لـ **SUI/USDT:USDT** مبني على Flask + CCXT ويعمل على Render.
يعتمد على Range Filter + Council + (اختياري) Golden Zones/Bookmap/Flow بنفس ستايل اللوج القديم.

## التشغيل المحلي
```bash
pip install -r requirements.txt
export FLASK_ENV=production
python -u main.py

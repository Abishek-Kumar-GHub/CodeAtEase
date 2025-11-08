# 🚀 Quick Setup Guide - CodeAtEase

## 📋 Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] GitHub account
- [ ] DeepSeek API account

---

## ⚡ Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CodeAtEase.git
cd CodeAtEase

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create `.env` File

```bash
# Copy example file
cp .env.example .env
```

Edit `.env` with your actual values:
```env
SECRET_KEY=use-openssl-rand-hex-32-to-generate
ALGORITHM=HS256

GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here
GITHUB_REDIRECT_URI=http://127.0.0.1:8000/auth/github/callback

DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
```

### 3. Create Templates Folder

```bash
mkdir templates
```

Move your HTML files:
- `index.html` → `templates/index.html` (Login page)
- `repo.html` → `templates/repo.html` (Repository selection)
- `aipage.html` → `templates/aipage.html` (Code editor)

### 4. Get API Keys

#### GitHub OAuth (Required)
1. Go to https://github.com/settings/developers
2. Click "New OAuth App"
3. Fill in:
   - **Name**: CodeAtEase
   - **Homepage**: http://127.0.0.1:8000
   - **Callback**: http://127.0.0.1:8000/auth/github/callback
4. Copy Client ID and Secret to `.env`

#### DeepSeek API (Required)
1. Go to https://platform.deepseek.com/
2. Sign up / Login
3. Go to "API Keys"
4. Click "Create API Key"
5. Copy key to `.env`

### 5. Run the Application

```bash
uvicorn main:app --reload --port 8000
```

### 6. Open in Browser

```
http://127.0.0.1:8000
```

---

## 🎯 File Structure

```
CodeAtEase/
├── main.py                 # ✅ FastAPI backend
├── requirements.txt        # ✅ Dependencies
├── .env                    # ✅ Your config (create this)
├── .env.example           # Template
├── .gitignore             # Git ignore
├── README.md              # Full documentation
├── SETUP.md               # This file
└── templates/             # ✅ HTML files
    ├── index.html         # Login page
    ├── repo.html          # Repository selection
    └── aipage.html        # Code editor
```

---

## 🔧 Common Issues

### Issue: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn main:app --reload --port 8001
```

### Issue: GitHub OAuth not working
- Check Client ID and Secret in `.env`
- Verify callback URL: `http://127.0.0.1:8000/auth/github/callback`
- Make sure not using `localhost` (use `127.0.0.1`)

### Issue: DeepSeek API error
- Verify API key in `.env`
- Check you have credits/quota
- Test key at https://platform.deepseek.com/

---

## 🧪 Test the Setup

### 1. Check Health Endpoint
```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T...",
  "ai_model": "DeepSeek Chat"
}
```

### 2. Check API Documentation
Open: http://127.0.0.1:8000/docs

You should see Swagger UI with all endpoints.

### 3. Test Login Flow
1. Go to http://127.0.0.1:8000/
2. Click "Sign in with GitHub"
3. Authorize on GitHub
4. Should redirect to repository page

---

## 📝 Environment Variables Explained

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key | Generate with `openssl rand -hex 32` |
| `ALGORITHM` | JWT algorithm | `HS256` (keep as is) |
| `GITHUB_CLIENT_ID` | OAuth app ID | `Iv1.abc123...` |
| `GITHUB_CLIENT_SECRET` | OAuth secret | `1a2b3c4d...` |
| `GITHUB_REDIRECT_URI` | OAuth callback | `http://127.0.0.1:8000/auth/github/callback` |
| `DEEPSEEK_API_KEY` | DeepSeek API | `sk-...` |

---

## 🚀 Production Deployment

### Update `.env` for Production:
```env
SECRET_KEY=use-strong-random-key
GITHUB_REDIRECT_URI=https://yourdomain.com/auth/github/callback
```

### Update GitHub OAuth:
- Homepage URL: `https://yourdomain.com`
- Callback URL: `https://yourdomain.com/auth/github/callback`

### Run in Production:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use Gunicorn:
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 💡 Tips

1. **Generate Secret Key**:
   ```bash
   openssl rand -hex 32
   ```

2. **Check Python Version**:
   ```bash
   python --version  # Should be 3.11+
   ```

3. **Update Dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **View Logs**:
   ```bash
   uvicorn main:app --reload --log-level debug
   ```

5. **Clear Browser Storage**:
   - Open DevTools (F12)
   - Console: `localStorage.clear()`
   - Refresh page

---

## 📞 Need Help?

- Check full documentation: `README.md`
- Open an issue on GitHub
- Check logs in terminal for errors

---

## ✅ Setup Complete!

If everything works:
- ✅ Server running on http://127.0.0.1:8000
- ✅ Can login with GitHub
- ✅ Repositories load dynamically
- ✅ AI analysis works with DeepSeek

**You're ready to code! 🎉**

---

## Screenshots
<img width="1920" height="1080" alt="Screenshot_2025-11-08_23_18_19" src="https://github.com/user-attachments/assets/beae4c1c-efcf-4981-9fca-d2d11dc32a07" />
- **Lander Page**
<img width="1920" height="1080" alt="Screenshot_2025-11-08_23_18_25" src="https://github.com/user-attachments/assets/3759974c-5a8c-46a4-b495-7204c9e59b6f" />\
- **Repository selection**
<img width="1920" height="1080" alt="Screenshot_2025-11-08_23_47_45" src="https://github.com/user-attachments/assets/755bf2f8-5bee-417e-8bbc-dfd103ac3180" />
- **CodeAtEaseDemo**


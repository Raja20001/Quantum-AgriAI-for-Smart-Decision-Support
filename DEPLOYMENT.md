# 🚀 Deployment & Hosting Guide: Quantum AgriAI

This document provides step-by-step instructions to host and deploy the **Quantum AgriAI** web application on **GitHub Pages** and **Vercel** for public research demonstration and publication.

---

## 🌟 Quick Overview of Deployment Targets

| Feature / Platform | 🐙 GitHub Pages | ▲ Vercel |
| :--- | :--- | :--- |
| **Best For** | Academic publication, permanent open-source hosting | High-speed global edge CDN, custom domain routing |
| **Cost** | 100% Free | 100% Free (Hobby tier) |
| **Zero-Config Setup** | ✅ Native (Root `/` or `/docs`) | ✅ Native via `vercel.json` |
| **Automated CI/CD** | ✅ GitHub Actions (`.github/workflows/deploy-pages.yml`) | ✅ Git-push automated deployments |
| **Custom Domain & SSL**| ✅ Free automatic HTTPS | ✅ Free automatic HTTPS & DNS |
| **Server Requirement** | Pure client-side static (No server needed) | Serverless / Static edge |

---

## 🐙 Method 1: Deploy to GitHub Pages (Recommended for Publication)

### Option A: 3-Click Setup via GitHub Repository Settings
1. Go to your GitHub repository:
   [https://github.com/Raja20001/Quantum-AgriAI-for-Smart-Decision-Support](https://github.com/Raja20001/Quantum-AgriAI-for-Smart-Decision-Support)
2. Click on **Settings** (top navigation tab).
3. In the left sidebar, click on **Pages** (under the "Code and automation" section).
4. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`.
   - **Branch**: Select `main`.
   - **Folder**: Select `/(root)` (or `/docs`).
5. Click **Save**.
6. Wait 1–2 minutes. Your website will be live at:
   ```
   https://raja20001.github.io/Quantum-AgriAI-for-Smart-Decision-Support/
   ```

---

### Option B: Automated GitHub Actions Workflow
A ready-to-run GitHub Actions workflow has already been configured in:
[`.github/workflows/deploy-pages.yml`](file:///.github/workflows/deploy-pages.yml)

1. Under your GitHub repository **Settings** → **Pages**:
   - Under **Source**, select **GitHub Actions**.
2. Whenever you push changes to the `main` branch, GitHub Actions will automatically validate, package, and publish the latest version of the dashboard.
3. Check the **Actions** tab in GitHub to see live deployment logs and the publication URL.

---

## ▲ Method 2: Deploy to Vercel (Edge CDN)

### Option A: Via Vercel Web Dashboard (1-Click Import)
1. Navigate to [https://vercel.com/new](https://vercel.com/new) and log in with your GitHub account.
2. Under **Import Git Repository**, search for `Quantum-AgriAI-for-Smart-Decision-Support` and click **Import**.
3. In the Project Configuration:
   - **Project Name**: `quantum-agri-ai` (or your preferred name)
   - **Framework Preset**: `Other` (Static HTML/JS)
   - **Root Directory**: `./` (leave default)
   - **Build Command**: *Leave blank* (Zero build step needed)
   - **Output Directory**: *Leave blank*
4. Click **Deploy**.
5. Within 15 seconds, Vercel will assign a production URL:
   ```
   https://quantum-agri-ai.vercel.app
   ```

---

### Option B: Via Vercel CLI (Command Line)
If you have Node.js installed, you can deploy directly from your local terminal:

```bash
# 1. Install the Vercel CLI globally
npm install -g vercel

# 2. Login to your Vercel account
vercel login

# 3. Deploy directly to production
vercel --prod
```
The CLI will read [`vercel.json`](file:///vercel.json) and output your live deployment link.

---

## 💻 Local Preview & Verification

To test and preview the website locally before publishing:

### Using Python:
```bash
# From the project root:
python -m http.server 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### Using Node / npx:
```bash
npx serve .
```

---

## 📁 Key Files Configured for Deployment

- [`index.html`](file:///index.html): Primary single-page application entry point at root.
- [`css/style.css`](file:///css/style.css): Modern dark glassmorphism stylesheet.
- [`js/engine.js`](file:///js/engine.js): Client-side Quantum-Classical inference engine (Gaussian likelihood + Agronomic rule engine).
- [`js/quantum.js`](file:///js/quantum.js): Interactive 4-qubit VQC circuit, statevector probabilities, and QAOA optimizer.
- [`js/explorer.js`](file:///js/explorer.js): Geo-spatial India agriculture explorer for 37 States/UTs.
- [`js/market.js`](file:///js/market.js): Interactive Mandi price trend chart with Bollinger Bands and MSP reference.
- [`data/`](file:///data/): Compact JSON datasets (`crops.json`, `india_agri.json`, `market.json`).
- [`docs/`](file:///docs/): Complete static mirror ready for GitHub Pages `/docs` mode.
- [`.nojekyll`](file:///.nojekyll): Prevents GitHub Pages from ignoring files or processing Jekyll.
- [`vercel.json`](file:///vercel.json): Vercel routing rules, clean URLs, and CORS security headers.
- [`.github/workflows/deploy-pages.yml`](file:///.github/workflows/deploy-pages.yml): GitHub Actions deployment automation.

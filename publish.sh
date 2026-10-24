#!/usr/bin/env bash
# 发布到 GitHub Pages。第一次：建仓库 → 推送 → 打开 Pages。之后：提交并推送更新。
#
#   bash publish.sh
#
# 页面地址：https://eric-s108.github.io/nem-weather-desk/
set -euo pipefail
cd "$(dirname "$0")"

OWNER="${OWNER:-Eric-S108}"
REPO="${REPO:-nem-weather-desk}"
OWNER_LC="$(printf '%s' "${OWNER}" | tr '[:upper:]' '[:lower:]')"
PAGE="https://${OWNER_LC}.github.io/${REPO}/"

if [ ! -f docs/index.html ]; then
  echo "还没有 docs/index.html，先跑：bash run_desk.sh"
  exit 1
fi

FIRST=0
if [ ! -d .git ]; then
  git init -q -b main
  FIRST=1
fi

# 没配过 git 身份时，只给这个仓库设一个 GitHub noreply 地址，不暴露真实邮箱
if ! git config user.email >/dev/null 2>&1; then
  git config user.name "${OWNER}"
  git config user.email "${OWNER}@users.noreply.github.com"
fi

git add -A
if [ "${FIRST}" = "1" ]; then
  git commit -q -F - <<'MSG' || true
NEM weather trading desk: benchmark ladder, battery economics, desk page

- Per-hour models (one per hour-of-day) instead of one shared model
- Every feature visible at the 24h decision time (price lags >= 26h,
  one-sided lead for weather and AEMO predispatch)
- Battery economics: 1 MW / 2 MWh, charge-before-discharge, throughput cost,
  NEM trading day; replaces l4_value
- Predispatch ingestion reads each run's publish time from the file header
- Desk page: four gates incl. >= 90 out-of-sample days; no local paths

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019zGMusjNSAWKYy79BLgFbr
MSG
else
  git commit -q -m "Update trading desk $(date -u +%Y-%m-%dT%H:%MZ)" || echo "没有新改动，直接推送"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -q origin main
  echo "已推送。页面约 1 分钟后更新：${PAGE}"
  exit 0
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "${OWNER}/${REPO}" --public --source=. --remote=origin --push
  gh api -X POST "repos/${OWNER}/${REPO}/pages" \
    -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 || true
  echo
  echo "仓库：https://github.com/${OWNER}/${REPO}"
  echo "页面：${PAGE} （第一次开启约 1–2 分钟后生效）"
  exit 0
fi

echo
echo "没有找到已登录的 gh 命令行，改用网页建仓库（只需要做一次）："
echo
echo "  1. 打开 https://github.com/new"
echo "     Repository name 填 ${REPO}，选 Public，下面三个勾都不要勾，点 Create repository"
echo
printf "  建好后按回车继续……"
read -r _
git remote add origin "https://github.com/${OWNER}/${REPO}.git"
git push -u origin main
echo
echo "  2. 打开 https://github.com/${OWNER}/${REPO}/settings/pages"
echo "     Source 选 Deploy from a branch；Branch 选 main，文件夹选 /docs；点 Save"
echo
echo "页面：${PAGE} （保存后约 1–2 分钟生效）"

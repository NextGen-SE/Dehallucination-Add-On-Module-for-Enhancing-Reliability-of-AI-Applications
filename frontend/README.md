# TruthLayer Frontend (Person 3)

Next.js dashboard: paste text -> Verify -> colour-coded claims with expandable evidence.

## Run
    npm install
    cp .env.local.example .env.local     # Windows: copy .env.local.example .env.local
    npm run dev                          # http://localhost:3000

## Mock vs real API
Edit `.env.local`:
- `NEXT_PUBLIC_USE_MOCK=true`  -> built-in fake data, no backend needed
- `NEXT_PUBLIC_USE_MOCK=false` -> calls `NEXT_PUBLIC_API_URL/verify`
Restart `npm run dev` after changing it.

## Tests
    npm test

## Structure
- `src/lib/api.js` - the only file that talks to the backend (+ response normalisation)
- `src/lib/validate.js` - word count / input validation (REQ-1, REQ-15)
- `src/components/` - NavBar, ClaimCard, ResultsView, Loader, ErrorBanner, Disclaimer
- `src/app/page.js` - main page
- `__tests__/` - unit/component tests

## Requirement mapping
REQ-1/15 input limit check, REQ-4 empty-claims message, REQ-7 confidence shown,
REQ-10 evidence passage + source id, REQ-16 colour coding, NFR-5 disclaimer.

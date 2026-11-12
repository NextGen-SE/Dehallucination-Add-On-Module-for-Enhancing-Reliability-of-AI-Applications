# TruthLayer Next.js dashboard. Build context: ./frontend
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
# NEXT_PUBLIC_* values are inlined into the browser bundle at build time, so
# they must be passed as build args. Only non-secret values belong here: the
# API key is typed into the UI by the user and never baked into the image.
# The URL is the backend as the *browser* sees it (published port on the host).
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_USE_MOCK=false
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_USE_MOCK=$NEXT_PUBLIC_USE_MOCK \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json ./
# Runtime deps only. Optional deps are the @next/swc-* compiler binaries
# (~280 MB), needed by `next build` above but not by `next start`.
RUN npm ci --omit=dev --omit=optional && npm cache clean --force
COPY --from=build --chown=node:node /app/.next ./.next
USER node
EXPOSE 3000
CMD ["node_modules/.bin/next", "start", "-H", "0.0.0.0", "-p", "3000"]

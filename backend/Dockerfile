# ----------------------------------------------------
# Stage 1: Build Stage
# ----------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY package*.json ./

# Install ALL dependencies (including devDependencies needed for build)
RUN npm ci

# Copy full application source code
COPY . .

# Run build step (compiles TypeScript/bundles app to /dist or /build)
RUN npm run build

# ----------------------------------------------------
# Stage 2: Production Runtime
# ----------------------------------------------------
FROM node:20-alpine AS runner

ENV NODE_ENV=production

WORKDIR /app

COPY package*.json ./

# Install ONLY production dependencies
RUN npm ci --omit=dev

# Copy built assets from builder stage
COPY --from=builder /app/dist ./dist

# Run as non-root user
USER node

EXPOSE 3000

# Run compiled output directly with node for better signal handling
CMD ["node", "dist/main.js"]

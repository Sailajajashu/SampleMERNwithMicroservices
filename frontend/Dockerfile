# Stage 1: Build React application
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files first
COPY package.json package-lock.json ./

# Install all dependencies
RUN npm ci

# Copy application source
COPY . .

# Build React application
RUN npm run build


# Stage 2: Serve React application using Nginx
FROM nginx:alpine

# Remove default nginx content
RUN rm -rf /usr/share/nginx/html/*

# Copy React production build
COPY --from=build /app/build /usr/share/nginx/html

# Expose HTTP port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

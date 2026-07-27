# Stage 1: Build React application
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build React application
RUN npm run build


# Stage 2: Serve React application using Nginx
FROM nginx:alpine

# Remove default Nginx files
RUN rm -rf /usr/share/nginx/html/*

# Copy React build files
COPY --from=build /app/build /usr/share/nginx/html

# Expose Nginx port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]

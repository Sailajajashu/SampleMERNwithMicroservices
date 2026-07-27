
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev || npm install --omit=deva
COPY . .
EXPOSE 5001
CMD ["node", "index.js"]

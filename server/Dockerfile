# build context: the repository root (docker compose build server)
FROM node:22-alpine
WORKDIR /app
COPY server/package.json server/package-lock.json ./server/
RUN cd server && npm ci --omit=dev
COPY server ./server
COPY db ./db
COPY web/snapshot.json ./web/snapshot.json
WORKDIR /app/server
ENV NODE_ENV=production PORT=3000
USER node
EXPOSE 3000
CMD ["node", "src/server.js"]

const CACHE = 'chromata-v1';
const ASSETS = [
  '/ColorQuiz/',
  '/ColorQuiz/index.html',
  '/ColorQuiz/manifest.json',
  '/ColorQuiz/icon-192.png',
  '/ColorQuiz/icon-512.png',
  '/ColorQuiz/fonts/cormorant-italic-400.ttf',
  '/ColorQuiz/fonts/cormorant-italic-500.ttf',
  '/ColorQuiz/fonts/cormorant-normal-400.ttf',
  '/ColorQuiz/fonts/cormorant-normal-500.ttf',
  '/ColorQuiz/fonts/cormorant-normal-600.ttf',
  '/ColorQuiz/fonts/dmsans-300.ttf',
  '/ColorQuiz/fonts/dmsans-400.ttf',
  '/ColorQuiz/fonts/dmsans-500.ttf',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});

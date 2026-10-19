// Like/dislike storage for the digest page (see templates/digest.html and
// core/interest.py). POST {urls: [...], v: 1|-1} records a vote per article
// URL; GET returns {url: v} for the pipeline's morning sync. Both need the
// x-key header to match the VOTE_KEY secret -- the site URL is public.
export async function onRequest({ request, env }) {
  if (!env.VOTE_KEY || request.headers.get("x-key") !== env.VOTE_KEY) {
    return new Response("unauthorized", { status: 401 });
  }
  if (request.method === "POST") {
    const { urls, v } = await request.json().catch(() => ({}));
    if (!Array.isArray(urls) || urls.length > 20 || (v !== 1 && v !== -1)) {
      return new Response("bad request", { status: 400 });
    }
    // The vote lives in the key's metadata so GET needs only list() calls.
    // 30-day expiry: the pipeline copies votes into history.db daily.
    await Promise.all(
      urls.map((u) => env.VOTES.put(String(u).slice(0, 500), "", { metadata: { v }, expirationTtl: 2592000 }))
    );
    return new Response("ok");
  }
  const votes = {};
  let cursor;
  do {
    const page = await env.VOTES.list({ cursor });
    for (const k of page.keys) votes[k.name] = k.metadata?.v;
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  return Response.json(votes);
}

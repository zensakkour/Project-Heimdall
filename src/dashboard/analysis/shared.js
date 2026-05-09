export function parseServerError(text) {
  if (!text) return "";
  const trimmed = String(text).trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed === "string") return parsed;
    if (parsed && typeof parsed === "object") {
      if (typeof parsed.detail === "string") return parsed.detail;
      if (Array.isArray(parsed.detail)) {
        const msgs = parsed.detail
          .map((item) => (item && item.msg ? item.msg : ""))
          .filter(Boolean);
        if (msgs.length) return msgs.join("; ");
      }
      if (typeof parsed.error === "string") return parsed.error;
      if (typeof parsed.message === "string") return parsed.message;
    }
  } catch {
    // Keep fallback raw text.
  }
  return trimmed.replace(/\s+/g, " ");
}

export function normalizeError(err) {
  if (!err) return "Unexpected error.";
  if (typeof err === "string") return parseServerError(err) || "Unexpected error.";
  if (err instanceof Error) return parseServerError(err.message) || "Unexpected error.";
  return parseServerError(String(err)) || "Unexpected error.";
}

export async function postForm(url, formData) {
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      body: formData,
      headers: (typeof formData === "string") ? {"Content-Type": "application/json"} : {},
    });
  } catch (err) {
    const msg = normalizeError(err);
    const networkLike =
      msg.toLowerCase() === "failed to fetch" ||
      msg.toLowerCase().includes("networkerror") ||
      msg.toLowerCase().includes("load failed");
    if (networkLike) {
      throw new Error(
        "Backend unreachable. Open the analysis UI through the Heimdall FastAPI server and verify /health/runtime is up."
      );
    }
    throw err;
  }
  if (!res.ok) {
    const text = await res.text();
    const message = parseServerError(text) || `Request failed (HTTP ${res.status}).`;
    throw new Error(message);
  }
  return res.json();
}

export function byId(id) {
  return document.getElementById(id);
}

export function formatUtcNowLabel(prefix) {
  const now = new Date();
  const iso = now.toISOString().replace("T", " ").slice(0, 19) + " UTC";
  return prefix ? `${prefix}${iso}` : iso;
}

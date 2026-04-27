const fallbackApiBaseUrl = "http://localhost:5000";

export const API_BASE_URL = (
  process.env.REACT_APP_API_BASE_URL || fallbackApiBaseUrl
).replace(/\/+$/, "");

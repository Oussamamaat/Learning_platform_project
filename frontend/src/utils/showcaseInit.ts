// Initialize showcase demo data if not already loaded
export async function initShowcaseData() {
  const STORAGE_KEY = "atlas_tutor.sessions.v1";

  // Check if we already have sessions
  const existing = localStorage.getItem(STORAGE_KEY);
  if (existing) {
    // Sessions already exist, don't override
    return;
  }

  try {
    // Try to load showcase data
    const response = await fetch("/showcase_data.json");
    if (response.ok) {
      const data = await response.json();
      if (data.sessions && Array.isArray(data.sessions)) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data.sessions));
      }
    }
  } catch (err) {
    // Silently fail - showcase data is optional
    console.log("Showcase data not available");
  }
}

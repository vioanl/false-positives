// Greek air data. Modelled: Copernicus CAMS Europe via the Open-Meteo air quality API.
// Station measurements (WAQI) come later.

export const CITIES = [
  { id:"thessaloniki", name:"Thessaloniki", lat:40.6403, lon:22.9439 },
  { id:"athens",       name:"Athens",       lat:37.9838, lon:23.7275 },
  { id:"corfu",        name:"Corfu",        lat:39.6243, lon:19.9217 },
  { id:"ioannina",     name:"Ioannina",     lat:39.6650, lon:20.8537 },
  { id:"kozani",       name:"Kozani",       lat:40.3006, lon:21.7890 }
];

// WHO short-term guideline values, µg/m³
export const WHO = { pm2_5:15, pm10:45, nitrogen_dioxide:25, ozone:100 };

export const POLL = {
  pm2_5:            { name:"PM2.5", color:"#ff2d55", size:4,  label:"PM2.5" },
  pm10:             { name:"PM10",  color:"#ff8c1a", size:7,  label:"PM10"  },
  nitrogen_dioxide: { name:"NO₂",   color:"#2f8bff", size:12, label:"NO2"   },
  ozone:            { name:"O₃",    color:"#b56bff", size:18, label:"O3"    }
};

export const ratio = (air, k) => (air[k] || 0) / WHO[k];

export async function getAir(city){
  const vars = Object.keys(WHO).join(",");
  const url = `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${city.lat}&longitude=${city.lon}&hourly=${vars}&timezone=auto&forecast_days=1`;
  const r = await fetch(url);
  if (!r.ok) throw new Error("air data unavailable");
  const j = await r.json();
  const times = j.hourly.time, now = new Date();
  let idx = 0;
  for (let i = 0; i < times.length; i++){ if (new Date(times[i]) <= now) idx = i; else break; }
  const out = { city: city.name, time: times[idx], source: "modelled" };
  for (const k of Object.keys(WHO)){
    const s = j.hourly[k] || [];
    let v = s[idx];
    for (let i = idx; i >= 0 && (v === null || v === undefined); i--) v = s[i];
    out[k] = (v === null || v === undefined) ? 0 : v;
  }
  return out;
}

export function fillCitySelect(select){
  select.innerHTML = CITIES.map(c => `<option value="${c.id}">${c.name}</option>`).join("");
}
export const cityById = id => CITIES.find(c => c.id === id) || CITIES[0];

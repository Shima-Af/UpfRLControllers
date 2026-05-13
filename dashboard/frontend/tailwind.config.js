/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // policy colors used across charts + summary cards
        policy: {
          ppo:       "#0ea5e9",
          random:    "#94a3b8",
          "always-dpdk": "#10b981",
          "always-usr":  "#f59e0b",
          threshold: "#a855f7",
        },
      },
    },
  },
  plugins: [],
};

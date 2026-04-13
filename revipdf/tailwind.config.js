/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./src/**/*.{html,js}",
  ],
  theme: {
    extend: {
      colors: {
        "surface-container-low": "#f8f1fe",
        "surface": "#ffffff",
        "tertiary-container": "rgba(0, 153, 255, 0.3)",
        "inverse-surface": "#170f23",
        "surface-dim": "#f7f5f8",
        "on-primary-container": "#4800ad",
        "secondary": "#665492",
        "surface-tint": "#4800ad",
        "outline-variant": "rgba(72, 0, 173, 0.1)",
        "secondary-fixed": "#e9ddff",
        "inverse-primary": "#d0bcff",
        "on-primary-fixed-variant": "#541db8",
        "surface-container-high": "#ede6f2",
        "on-tertiary": "#ffffff",
        "on-primary": "#ffffff",
        "on-tertiary-fixed": "#370e00",
        "tertiary": "#0099ff",
        "on-secondary-fixed": "#210e4a",
        "on-secondary-container": "#584683",
        "error": "#ba1a1a",
        "tertiary-fixed-dim": "#ffb599",
        "on-surface": "#1d1a23",
        "surface-container": "#f2ebf8",
        "primary": "#4800ad",
        "surface-container-highest": "#e7e0ed",
        "on-primary-fixed": "#23005c",
        "surface-container-lowest": "#ffffff",
        "error-container": "#ffdad6",
        "outline": "#7a7485",
        "on-surface-variant": "#494454",
        "on-tertiary-fixed-variant": "#7c2e09",
        "on-secondary": "#ffffff",
        "on-tertiary-container": "#006bb3",
        "surface-bright": "#ffffff",
        "on-error": "#ffffff",
        "inverse-on-surface": "#f5eefb",
        "secondary-container": "#cdb9fe",
        "on-error-container": "#93000a",
        "secondary-fixed-dim": "#d0bcff",
        "on-background": "#1d1a23",
        "tertiary-fixed": "#ffdbce",
        "on-secondary-fixed-variant": "#4e3c78",
        "background": "#f7f5f8",
        // Keeping original colors just in case
        "background-light": "#f7f5f8",
        "background-dark": "#170f23",
        "grammar-highlight": "rgba(255, 242, 0, 0.4)",
        "semantic-highlight": "rgba(0, 153, 255, 0.3)",
      },
      fontFamily: {
        "headline": ["Inter"],
        "body": ["Inter"],
        "label": ["Inter"],
        "inter": ["Inter"],
        "display": ["Inter"]
      },
      borderRadius: {"DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px"},
      keyframes: {
        'gradient-x': {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          },
        },
      },
      animation: {
        'gradient-x': 'gradient-x 3s ease infinite',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries'),
  ],
}

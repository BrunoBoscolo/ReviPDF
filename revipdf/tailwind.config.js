/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./src/**/*.{html,js}",
  ],
  theme: {
    extend: {
      colors: {
          "primary": "#4800ad",
          "background-light": "#f7f5f8",
          "background-dark": "#170f23",
          "grammar-highlight": "rgba(255, 242, 0, 0.4)",
          "semantic-highlight": "rgba(0, 153, 255, 0.3)",
      },
      fontFamily: {
          "display": ["Inter"]
      },
      borderRadius: {"DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px"},
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries'),
  ],
}

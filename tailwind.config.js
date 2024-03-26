/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js,jsx}"],
  theme: {
    extend: {
      colors: {
        "success-25": "#F6FEF9",
        "success-200": "#A6F4C5",
        "success-400": "#32D583",
        "success-600": "#039855",
        "error-200": "#FECDCA",
        "error-400": "#FEF3F2",
        "error-50": "#FEF3F2",
        "b-1000": "#073032",
        "error-600": "#D92D20",
        "a-50": "#EEF5FF",
        "gray-400": "#98A2B3",
      },
    },
  },
  plugins: [],
};

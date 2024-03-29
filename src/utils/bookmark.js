// Function to get the bookmarked number variable from localStorage
export const getBookmarked = () => {
  const bookmarked = localStorage.getItem("bookmarked");
  return bookmarked ? parseInt(bookmarked) : 0;
};

// Function to update the bookmarked number variable in localStorage
export const setBookmarkedToStorage = (value) => {
  localStorage.setItem("bookmarked", value.toString());
};

// Function to initialize the bookmarked number variable in localStorage if it doesn't exist
export const initBookmarked = () => {
  if (!localStorage.getItem("bookmarked")) {
    setBookmarkedToStorage(0);
  }
};

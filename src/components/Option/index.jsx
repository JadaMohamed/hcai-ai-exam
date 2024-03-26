// Option.js
import React from "react";

const Option = ({
  isCorrectAnswer,
  isCollapsed,
  number,
  label,
  i,
  isChecked,
  isQCM,
  handleOptionSelect,
}) => {
  const handleCheckboxChange = (event) => {
    handleOptionSelect(label[0], event.target.checked);
  };

  return (
    <div
      className={`flex gap-3 duration-300 ${
        isCorrectAnswer && !isCollapsed
          ? " bg-[#0398552f] "
          : isChecked && !isCorrectAnswer && !isCollapsed
          ? " bg-[#D92D202f] "
          : ""
      }`}
      key={i}
    >
      <input
        type={isQCM ? "checkbox" : "radio"}
        name={"ques" + number}
        id={i + String(number) + "q"}
        value={label[0]}
        checked={isChecked}
        onChange={handleCheckboxChange}
      />
      <label htmlFor={i + String(number) + "q"}>{label}</label>
    </div>
  );
};

export default Option;

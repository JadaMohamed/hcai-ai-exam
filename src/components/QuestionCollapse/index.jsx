// QuestionCollapse.js
import React, { useEffect, useState } from "react";
import Option from "../Option";

const QuestionCollapse = ({
  number,
  type,
  question,
  options,
  answers,
  parentCollapse,
  handleBookmark,
  isBookmarked,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(parentCollapse);
  const [selectedOptions, setSelectedOptions] = useState([]);

  useEffect(() => {
    setIsCollapsed(parentCollapse);
  }, [parentCollapse]);
  const handleToggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  const handleOptionSelect = (optionValue, isChecked) => {
    if (type === "qcm" ? true : false) {
      // For QCM (multiple choice), toggle the selected option
      if (isChecked) {
        setSelectedOptions([...selectedOptions, optionValue]);
      } else {
        setSelectedOptions(
          selectedOptions.filter((opt) => opt !== optionValue)
        );
      }
    } else {
      // For single choice, directly set the selected option
      setSelectedOptions([optionValue]);
    }
  };

  return (
    <div className="relative group" id={"question-" + number}>
      <div className="text-gray-100 m-auto max-w-xl w-full bg-[#ffffff0a] border border-[#ffffff13] relative rounded-lg">
        <button
          className={`bg-[#ffffff0a]  ${
            isBookmarked ? "" : "opacity-0 group-hover:opacity-100 "
          } absolute left-[-48px] p-2 rounded-lg hover:bg-[#ffffff13] duration-300`}
          title="Shuffle questions"
          onClick={() => handleBookmark(number)}
        >
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M5 7.8C5 6.11984 5 5.27976 5.32698 4.63803C5.6146 4.07354 6.07354 3.6146 6.63803 3.32698C7.27976 3 8.11984 3 9.8 3H14.2C15.8802 3 16.7202 3 17.362 3.32698C17.9265 3.6146 18.3854 4.07354 18.673 4.63803C19 5.27976 19 6.11984 19 7.8V21L12 17L5 21V7.8Z"
              className={` ${
                isBookmarked ? "fill-gray-300" : "stroke-gray-300"
              }`}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
        <div className="p-4">
          <span className="text-[#ffffff13] text-8xl font-bold absolute top-0 right-0 z-0 select-none">
            {number}
          </span>
          <h1 className="text-white font-semibold">{question}</h1>
          <div className="text-gray-300 mt-4 flex flex-col gap-1">
            {options.map((e, i) => (
              <Option
                key={i}
                isCorrectAnswer={answers.includes(e.charAt(0))}
                label={e}
                isCollapsed={isCollapsed}
                number={number}
                i={i}
                isChecked={selectedOptions.includes(e[0])}
                isQCM={type === "qcm" ? true : false}
                handleOptionSelect={handleOptionSelect}
              />
            ))}
          </div>
          <div className="flex justify-end">
            <button
              type="button"
              className="text-sm px-3 py-2 mt-2 bg-[#ffffff13] font-normal z-10 hover:bg-[#ffffff0a] duration-300 rounded-lg"
              onClick={handleToggleCollapse}
            >
              {isCollapsed ? "Show answer" : "Hide answer"}
            </button>
          </div>
          <div
            className={`italic transition-all duration-300 text-success-600 ${
              isCollapsed ? "max-h-0 opacity-0" : "max-h-[500px] opacity-100"
            }`}
          >
            Correct Answer :{" "}
            <span className="font-semibold">
              {answers.map((e, i) => (
                <span key={i}>{e} </span>
              ))}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuestionCollapse;

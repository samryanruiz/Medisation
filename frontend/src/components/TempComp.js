import React, { useEffect, useRef, useState } from 'react';
import "./TempComp.css";

const gifImage = process.env.PUBLIC_URL + '/tempAnimation.gif'; // Import the GIF file

const TempComp = ({
  aNIMATIONONHOWTOUSETHETEM,
  onGroupContainer1Click,
}) => {
  const gifRef = useRef(null);
  const [loaded, setLoaded] = useState(false); // Flag to track whether the GIF has been loaded

  useEffect(() => {
    let observer;

    const handleIntersection = (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          // Start loading the GIF and reset it to the starting frame if it's the first load
          if (!loaded && gifRef.current) {
            gifRef.current.src = gifImage;
            gifRef.current.currentTime = 0;
            setLoaded(true); // Set the loaded flag to true
          }
        } else {
          // Stop loading the GIF when it goes out of view
          if (gifRef.current) {
            gifRef.current.src = '';
          }
        }
      });
    };

    if (typeof IntersectionObserver !== 'undefined' && gifRef.current) {
      observer = new IntersectionObserver(handleIntersection);
      observer.observe(gifRef.current);
    }

    return () => {
      if (observer && gifRef.current) {
        observer.unobserve(gifRef.current);
      }
    };
  }, [loaded]); // Add loaded as a dependency to update the effect when it changes

  return (
    <div className="frame-parent7">
      <div className="vector-parent">
        {/* Attach the ref to the image element */}
        <img ref={gifRef} alt="tempAnimation.gif" style={{ width: '60%', height: '100%' }} />
        
        <img className="rectangle-icon" alt="" src="/rectangle-9.svg" />  
      </div>
      <div className="group-div" onClick={onGroupContainer1Click}>
        <div className="frame-child4" />
        <h1 className="start">START</h1>
      </div>
    </div>
  );
}

export default TempComp;

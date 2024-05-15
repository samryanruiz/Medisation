import "./FrameComponent1.css";

const FrameComponent1 = ({
  onHome2StreamlineCoresvgClick,
  onHOMETextClick,
}) => {
  return (
    <header className="frame-header">
      <div className="frame-child3" />
      <div className="speech-language-therapy-container">
        <button className="speech-language-therapy2">
          <img
            className="logonew"
            alt=""
            src="/logonew.png"
            onClick={onHome2StreamlineCoresvgClick}
          />
        </button>
        <div className="health-kiosk-title">
          <h1 className="health-kiosk2" onClick={onHOMETextClick}>MediSation</h1>
        </div>
      </div>
      <div className="frame-wrapper4">
        <div className="frame-parent6">
          <div className="home-2-streamline-coresvg-container">
            <img
              className="home-2-streamline-coresvg-icon1"
              loading="lazy"
              alt=""
              src="/home2streamlinecoresvg.svg"
              
            />
          </div>
          <div className="home-button">
            <b className="home1" onClick={onHOMETextClick}>
              HOME
            </b>
          </div>
          <img className="about-icon2" loading="lazy" alt="" src="/about.svg" />
          <div className="about-title">
            <b className="about2">ABOUT</b>
          </div>
        </div>
      </div>
    </header>
  );
};

export default FrameComponent1;

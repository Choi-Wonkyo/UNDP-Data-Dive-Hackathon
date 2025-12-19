const Back = () => {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 0,
        overflow: "hidden",
        pointerEvents: "none", // 클릭 막힘 방지
      }}
    >
      {/* 첫 번째 원 */}
      <div
        style={{
          position: "absolute",
          top: "38.83px",
          left: "-549.78px",
          width: "1258.453px",
          height: "1223.605px",
          transform: "rotate(175.68deg)",
          borderRadius: "1258.453px",
          background:
            "conic-gradient(from 180deg at 50% 50%, #004899 107.3deg, #95B9FF 233.62deg, #D795FF 310.5deg)",
          filter: "blur(145px)",
        }}
      />

      {/* 두 번째 원 */}
      <div
        style={{
          position: "absolute",
          top: "-413.08px",
          left: "-1127.26px",
          width: "1233.78px",
          height: "1533.585px",
          transform: "rotate(-137.443deg)",
          borderRadius: "1533.585px",
          background:
            "conic-gradient(from 180deg at 50% 50%, #95B9FF 233.62deg, #D795FF 310.5deg)",
          filter: "blur(120px)",
          mixBlendMode: "multiply",
        }}
      />

      {/* 세 번째 원 */}
      <div
        style={{
          position: "absolute",
          top: "51.49px",
          left: "-1241px",
          width: "1071.631px",
          height: "1254.474px",
          transform: "rotate(-56.448deg)",
          borderRadius: "1254.474px",
          background:
            "conic-gradient(from 180deg at 50% 50%, #004899 0deg, #004899 70.5deg, #38608C 162.37deg, #95B9FF 233.62deg, #BA95FF 310.5deg)",
          filter: "blur(120px)",
          mixBlendMode: "overlay",
        }}
      />
    </div>
  );
};

export default Back;

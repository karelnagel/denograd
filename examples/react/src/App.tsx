// @deno-types="@types/react"
import { useState } from "react";

function App() {
  const [count, setCount] = useState(0);

  return (
    <>
      <h1>Vite + React</h1>
      <button onClick={() => setCount((count) => count + 1)}>count is {count}</button>
    </>
  );
}

export default App;

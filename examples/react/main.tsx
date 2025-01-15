import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { useState } from "react";
import { Tensor} from "@denograd/denograd";

const LIST = [1, 2, 3, 4, 5];
const App = () => {
  
  const [res, setRes] = useState(LIST);
  return (
    <>
      <h1>
        Multiply {JSON.stringify(res)} by {JSON.stringify(LIST)}
      </h1>
      <button onClick={async () => setRes(await new Tensor(res).mul(new Tensor(LIST)).tolist())}>calc</button>
      <p>res: {JSON.stringify(res)}</p>
    </>
  );
};

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);

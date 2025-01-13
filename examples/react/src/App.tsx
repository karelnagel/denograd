import { useState } from "react";
import { Tensor } from "@denograd/denograd";

export const App = () => {
  const [res, setRes] = useState();
  return (
    <>
      <h1>Vite + React</h1>
      <button onClick={async () => setRes(await new Tensor([4]).mul(new Tensor([5])).tolist())}>calc</button>
      <p>res: {JSON.stringify(res)}</p>
    </>
  );
};

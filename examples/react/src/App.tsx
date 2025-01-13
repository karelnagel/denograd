import { useState } from "react";
import { Tensor } from "@denograd/denograd";

export const App = () => {
  const [res, setRes] = useState();
  const calc = () =>
    new Tensor([4])
      .add(new Tensor([5]))
      .tolist()
      .then((x) => setRes(x));
  return (
    <>
      <h1>Vite + React</h1>
      <button onClick={calc}>calc</button>
      <p>res: {JSON.stringify(res)}</p>
    </>
  );
};

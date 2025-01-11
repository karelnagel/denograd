import { useEffect, useState } from "react";
import { Tensor } from "../../../denograd/mod.ts";

export const App = () => {
  const [res, setRes] = useState();
  useEffect(() =>
    new Tensor([4])
      .add(new Tensor([5]))
      .tolist()
      .then((res) => setRes(res))
  );
  return (
    <>
      <h1>Vite + React</h1>
      4+5={res}
    </>
  );
};

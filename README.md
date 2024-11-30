# Denograd - Tinygrad rewrite in Typescript

## Why rewrite in typescript?

1. I think it's a better language
2. Even if 1. isn't objectively correct, there are 17.5M(source ChatGPT) JS developers that might need to write AI models in the future and rn the only good option is to use python. If Karpathy's
   [software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35) thesis is correct then either 1) there needs to be good JS ML framework or 2) everyone should start using python - imo 1. is more realistic future.
3. I wrote micrograd in TS and it was more than 10x faster than the python version (I know that tinygrad doesn't actually run the models in python, but still I think faster language might have some benefits)
4. All the JS web frameworks would have easier way to integrate models(I know you can generate WebGPU code from tinygrad, but still it's more complex than just having the model in TS)
5. I don't know that much about ML (and hardware and lower level programming), so I can learn
6. TS isn't much more verbose than python, so should be able to have similar line count to python. Sure python operator overloading is nice, but I think it's not that big of a deal.
7. Deno has Jupyter notebook support
8. Tinygrad should be under 10 000 lines, so it shouldn't be that hard to convert it 😅. Writing tinygrad = very hard, converting it to another language = shouldn't be that hard
9. The bar for best python ML lib is really high now, but I think for JS it's much lower, best one should be tensorflowJS. I think tinygrad will win in general and I like their philosophy.

## Roadmap

1. Get beautiful_mnist working with CLANG, TS or METAL runtime (not sure which one will be the easiest, but will find out). Write a lot of tests to make sure that the tinygrad implementation works the same way. Keep the file structure similar to
   python.
2. Convert other backends and features that are included in tinygrad core.
3. Keep the project in sync, tinygrad has ~15 PRs merged daily, so maybe have AI create PR for every tiny PR and let human edit and merge it.
4. Make the TS version faster and better than python
5. tinygrad realizes their mistake, drops python and start using this TS version (only partially joking)

## Money?

idk, I think that JS frameworks, runtimes and VCs should be interested in having competitive JS ML framework. Also when tinygrad has CLOUD=1 ready and is making chips then they should also be interested in getting JS developers to use their products.

-- these are just some notes for myself so I won't forget why I started doing this

`git rm $(git diff --name-only --diff-filter=U)`

# Third-Party Attribution

## LMCache

ConfKV is built on top of LMCache.

- Upstream project: LMCache
- Upstream repository: `LMCache/LMCache`
- ConfKV LMCache fork: `MountVeil/LMCache`
- Upstream baseline:
  `3031f71e66f8872f8c763544e6ad4a654e566629`
- License: Apache License 2.0

The ConfKV-modified LMCache runtime is included through the
`LMCache/` Git submodule.

The superproject pins the submodule to an exact commit for
reproducibility. Development of ConfKV-specific LMCache changes is
maintained on the `confkv` branch of `MountVeil/LMCache`.

ConfKV-specific additions outside LMCache are maintained in the
ConfKV repository itself.

See `MODIFICATIONS.md` for details.

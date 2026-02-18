#!/bin/bash
add-nix
export NIXPKGS_ALLOW_UNFREE=1
nix profile install nixpkgs#dua
nix profile install nixpkgs#pbzip2
nix profile install nixpkgs#gh
nix profile install nixpkgs#nodejs

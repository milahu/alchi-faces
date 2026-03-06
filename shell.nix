{
  pkgs ? import <nixpkgs> { }
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages (pp: with pp; [
      pillow
      # pillow-avif-plugin
      imagehash
      scikit-image
      numpy
    ]))
  ];
}

{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; config.cudaSupport = true; };

    libPath = with pkgs; pkgs.lib.makeLibraryPath [
      stdenv.cc.cc.lib
      linuxPackages.nvidia_x11
    ];
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python313
        cudaPackages.cudatoolkit
      ];

      shellHook = ''
        export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
        export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}

        if [ ! -d ".venv" ]; then
          python3 -m venv .venv
          source .venv/bin/activate

          pip install -r requirements.txt
        else
          source .venv/bin/activate
        fi
      '';
    };
  };
}
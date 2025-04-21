{ pkgs ? import <nixpkgs> {} }:
let

  pkbar = pkgs.python312Packages.buildPythonPackage rec {
    pname = "pkbar";
    version = "0.5";

    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "sha256-Omw4lojPpwyEQzFx7OcQmMf3GB3juoN1AZ3lcTUB7xU=";
    };

    # tell Nix to inject setuptools_scm rather than pip-fetch it
    nativeBuildInputs = [ pkgs.python312Packages.setuptools_scm ];

    # if pkbar itself needs other packages at runtime, list them here:
    propagatedBuildInputs = [ ]; 
    doCheck = false; # optionally skip tests
  };
in
  pkgs.mkShell {
    buildInputs = [
      pkgs.python312Packages.numpy
      pkgs.python312Packages.torch-bin
      pkgs.python312Packages.wandb
      pkgs.python312Packages.pip
      pkgs.python312Packages.pandas
      pkgs.python312Packages.pathos
      pkgs.python312Packages.torchvision-bin
      pkgs.python312Packages.einops
      pkbar
    ];
    # TODO: adding nvidia-x11/lib to LD_LIBRARY_PATH makes wandb correctly collect gpu metrics, but it's not
    # machine-agnostic
    shellHook = ''
      export LD_LIBRARY_PATH=/nix/store/kbmp6wl3a0h0gy25xxdblc892rp69qpm-nvidia-x11-570.133.07-6.14.1/lib:$LD_LIBRARY_PATH
      # if [ ! -d .venv ]; then
      #   echo "Creating virtualenv in .venv"
      #   python -m venv .venv
      # fi
      # echo "Activate python environment, then run: pip install -e ."
    ''; 
  }
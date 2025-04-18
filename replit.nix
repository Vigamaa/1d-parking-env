{pkgs}: {
  deps = [
    pkgs.ffmpeg-full
    pkgs.python311Packages.gymnasium
    pkgs.python312Packages.gymnasium
    pkgs.postgresql
    pkgs.openssl
  ];
}

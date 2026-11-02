#!/usr/bin/env bash
# scripts/build_custom_ffmpeg.sh
# Builds a minimal, static, audio-only FFmpeg & FFprobe for the AuraMaster Python library.

set -euo pipefail

FFMPEG_VERSION="6.1.1"
LAME_VERSION="3.100"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_BIN_DIR="${PROJECT_ROOT}/auramaster/bin"
BUILD_DIR="${PROJECT_ROOT}/build_ffmpeg_workspace"
PREFIX_DIR="${BUILD_DIR}/prefix"
CPU_CORES="$(nproc || echo 2)"

echo "============================================================"
echo " AuraMaster Custom Audio-Only FFmpeg Builder (Linux)"
echo "============================================================"
echo "Build workspace : ${BUILD_DIR}"
echo "Target bin dir  : ${TARGET_BIN_DIR}"
echo "Detected cores  : ${CPU_CORES}"

for cmd in gcc g++ make curl tar nasm pkg-config strip; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[ERROR] Required build tool '$cmd' is not installed." >&2
        echo "Install via: sudo apt-get install build-essential curl tar nasm pkg-config" >&2
        exit 1
    fi
done

mkdir -p "${BUILD_DIR}" "${PREFIX_DIR}" "${TARGET_BIN_DIR}"
cd "${BUILD_DIR}"

# 1. Build static libmp3lame
if [ ! -f "${PREFIX_DIR}/lib/libmp3lame.a" ]; then
    echo "[1/3] Downloading and building static libmp3lame ${LAME_VERSION}..."
    curl -L --retry 3 "https://sourceforge.net/projects/lame/files/lame/${LAME_VERSION}/lame-${LAME_VERSION}.tar.gz/download" -o "lame-${LAME_VERSION}.tar.gz"
    tar -xzf "lame-${LAME_VERSION}.tar.gz"
    cd "lame-${LAME_VERSION}"
    ./configure \
        --prefix="${PREFIX_DIR}" \
        --enable-static \
        --disable-shared \
        --enable-nasm \
        --disable-frontend \
        --disable-decoder
    make -j"${CPU_CORES}"
    make install
    cd "${BUILD_DIR}"
else
    echo "[1/3] Static libmp3lame already built. Skipping."
fi

# 2. Download FFmpeg
if [ ! -d "ffmpeg-${FFMPEG_VERSION}" ]; then
    echo "[2/3] Downloading FFmpeg ${FFMPEG_VERSION}..."
    curl -L --retry 3 "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o "ffmpeg-${FFMPEG_VERSION}.tar.xz"
    tar -xJf "ffmpeg-${FFMPEG_VERSION}.tar.xz"
fi

# 3. Configure and Compile Audio-Only FFmpeg
echo "[3/3] Configuring stripped Audio-Only FFmpeg..."
cd "ffmpeg-${FFMPEG_VERSION}"

export PKG_CONFIG_PATH="${PREFIX_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
CFLAGS="-I${PREFIX_DIR}/include -O3 -ffast-math -fPIC"
LDFLAGS="-L${PREFIX_DIR}/lib -static-libgcc -static-libstdc++"

./configure \
    --prefix="${PREFIX_DIR}" \
    --extra-cflags="${CFLAGS}" \
    --extra-ldflags="${LDFLAGS}" \
    --extra-libs="-lm -lpthread" \
    --pkg-config-flags="--static" \
    --enable-static \
    --disable-shared \
    --enable-small \
    --enable-pthreads \
    --disable-doc \
    --disable-htmlpages \
    --disable-manpages \
    --disable-podpages \
    --disable-txtpages \
    --disable-debug \
    --disable-network \
    --disable-autodetect \
    --disable-everything \
    --disable-avdevice \
    --disable-swscale \
    --disable-postproc \
    --disable-ffplay \
    --enable-ffmpeg \
    --enable-ffprobe \
    --enable-avcodec \
    --enable-avformat \
    --enable-avutil \
    --enable-avfilter \
    --enable-swresample \
    --enable-libmp3lame \
    --enable-protocol="file,pipe,data" \
    --enable-demuxer="wav,flac,mp3,ogg,mov,aac,aiff,au,caf,ape,tta,wv,pcm*" \
    --enable-muxer="wav,flac,mp3,ogg,ipod,mp4,adts,aiff,au,caf,tta,wv,null,pcm*" \
    --enable-parser="aac,aac_latm,ac3,flac,mpegaudio,opus,vorbis" \
    --enable-decoder="pcm*,flac,mp3*,aac*,ac3,eac3,alac,vorbis,opus,wma*,ape,tta,wavpack,dsd*" \
    --enable-encoder="pcm*,flac,libmp3lame,aac,alac,ac3,eac3,vorbis,opus,tta,wavpack" \
    --enable-filter="abuffer,abuffersink,afifo,aformat,amix,amerge,anull,apad,aperms,aresample,asetnsamples,asetpts,asetrate,asettb,ashowinfo,asidedata,asplit,astats,atrim,acopy,acompressor,acontrast,acrusher,acue,adeclick,adeclip,adelay,adenorm,aderivative,adrc,adynamicequalizer,adynamicsmooth,aecho,aemphasis,aeval,aexciter,afade,afftdn,afftfilt,afir,agate,aiir,aintegral,alimiter,allpass,aloop,ametadata,anequalizer,anlmdn,anlms,aphaser,apsyclip,apulsator,arealtime,asoftclip,aspectralstats,asubboost,asubcut,asupercut,asuperpass,asuperstop,atempo,atilt,axcorrelate,bandpass,bandreject,bass,biquad,channelmap,channelsplit,chorus,compand,compensationdelay,crossfeed,crystalizer,dcshift,deesser,dialoguenorm,drmeter,dynaudnorm,earwax,ebur128,equalizer,extrastereo,firequalizer,flanger,haas,hdcd,headphone,highpass,highshelf,join,loudnorm,lowpass,lowshelf,mcompand,pan,replaygain,sidechaincompress,sidechaingate,silencedetect,silenceremove,speechnorm,stereotools,stereowiden,superequalizer,surround,treble,tremolo,vibrato,virtualbass,volume,volumedetect"

echo "Compiling custom FFmpeg..."
make -j"${CPU_CORES}"
make install

# Strip symbols and copy to Python package bin directory
strip --strip-all "${PREFIX_DIR}/bin/ffmpeg"
strip --strip-all "${PREFIX_DIR}/bin/ffprobe"

cp -f "${PREFIX_DIR}/bin/ffmpeg" "${TARGET_BIN_DIR}/ffmpeg"
cp -f "${PREFIX_DIR}/bin/ffprobe" "${TARGET_BIN_DIR}/ffprobe"
chmod 755 "${TARGET_BIN_DIR}/ffmpeg" "${TARGET_BIN_DIR}/ffprobe"

echo "============================================================"
echo " Build Complete!"
echo " Custom binaries installed to: ${TARGET_BIN_DIR}"
ls -lh "${TARGET_BIN_DIR}"
echo "============================================================"
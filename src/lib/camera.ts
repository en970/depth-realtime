/**
 * Camera acquisition.
 *
 * Constraints are always `ideal`, never `exact`: an exact facingMode throws
 * OverconstrainedError on desktops, which have no rear camera. Device labels are
 * empty until permission has been granted, so the picker can only be populated
 * after the first successful stream.
 *
 * The video element must stay in the DOM and visible. WebKit refuses to deliver
 * frames to a video that is display:none or detached, so "hide the video and
 * only show the depth pane" is not an available refactor on iOS.
 */

export type CameraFailure =
  | "denied"
  | "not-found"
  | "in-use"
  | "insecure"
  | "unsupported"
  | "unknown";

export interface CameraError {
  kind: CameraFailure;
  title: string;
  detail: string;
}

export interface CameraDevice {
  id: string;
  label: string;
}

function describe(error: unknown): CameraError {
  const name = error instanceof DOMException ? error.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        kind: "denied",
        title: "Camera access was blocked",
        detail:
          "Allow camera access for this site in the browser address bar, then try again.",
      };
    case "NotFoundError":
    case "DevicesNotFoundError":
      return {
        kind: "not-found",
        title: "No camera detected",
        detail: "Connect a camera and try again.",
      };
    case "NotReadableError":
    case "TrackStartError":
      return {
        kind: "in-use",
        title: "The camera is busy",
        detail:
          "Another application is holding the camera. Close it and try again.",
      };
    default:
      return {
        kind: "unknown",
        title: "The camera could not be started",
        detail: error instanceof Error ? error.message : String(error),
      };
  }
}

interface CameraHandlers {
  onInterrupted: () => void;
  onResumed: () => void;
  onEnded: () => void;
}

export class Camera {
  private stream: MediaStream | null = null;
  private deviceId: string | null = null;
  private readonly video: HTMLVideoElement;
  private readonly handlers: CameraHandlers;

  constructor(video: HTMLVideoElement, handlers: CameraHandlers) {
    this.video = video;
    this.handlers = handlers;
  }

  static supported(): boolean {
    return Boolean(navigator.mediaDevices?.getUserMedia);
  }

  static insecureContext(): boolean {
    return !window.isSecureContext;
  }

  get active(): boolean {
    return this.stream !== null;
  }

  get resolution(): { width: number; height: number } | null {
    if (!this.video.videoWidth) return null;
    return { width: this.video.videoWidth, height: this.video.videoHeight };
  }

  get currentDeviceId(): string | null {
    return this.deviceId;
  }

  async start(deviceId?: string): Promise<CameraError | null> {
    if (Camera.insecureContext()) {
      return {
        kind: "insecure",
        title: "A secure context is required",
        detail:
          "Camera access needs HTTPS or localhost. Open the site over https://.",
      };
    }
    if (!Camera.supported()) {
      return {
        kind: "unsupported",
        title: "This browser cannot open a camera",
        detail:
          "navigator.mediaDevices is unavailable. In-app browsers often block it; open the page in Safari or Chrome.",
      };
    }

    // iOS cannot capture from two cameras at once: release the old stream first.
    this.stop();

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: deviceId
        ? { deviceId: { ideal: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : {
            facingMode: { ideal: "user" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 },
          },
    };

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      if (error instanceof DOMException && error.name === "OverconstrainedError") {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
        } catch (retry) {
          return describe(retry);
        }
      } else {
        return describe(error);
      }
    }

    this.stream = stream;
    this.video.srcObject = stream;

    const track = stream.getVideoTracks()[0];
    this.deviceId = track?.getSettings().deviceId ?? deviceId ?? null;
    track?.addEventListener("mute", this.handlers.onInterrupted);
    track?.addEventListener("unmute", this.handlers.onResumed);
    track?.addEventListener("ended", this.handlers.onEnded);

    await new Promise<void>((resolve) => {
      if (this.video.readyState >= 2) return resolve();
      this.video.addEventListener("loadedmetadata", () => resolve(), { once: true });
    });

    try {
      await this.video.play();
    } catch {
      // Autoplay can still be refused; the frame loop tolerates a paused video
      // and playback resumes on the next user gesture.
    }
    return null;
  }

  stop(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  /** Labels are only populated once permission exists, so call this after start. */
  async devices(): Promise<CameraDevice[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const all = await navigator.mediaDevices.enumerateDevices();
    return all
      .filter((device) => device.kind === "videoinput")
      .map((device, index) => ({
        id: device.deviceId,
        label: device.label || `Camera ${index + 1}`,
      }));
  }

  /** Re-issues play() after the tab returns to the foreground. */
  async resume(): Promise<void> {
    if (!this.stream) return;
    try {
      await this.video.play();
    } catch {
      /* ignored: requires a gesture on some platforms */
    }
  }
}

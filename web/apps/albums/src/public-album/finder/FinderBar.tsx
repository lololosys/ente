import { downloadManager } from "@/public-album/download/services/download-manager";
import CloseIcon from "@mui/icons-material/Close";
import FaceIcon from "@mui/icons-material/Face";
import SearchIcon from "@mui/icons-material/Search";
import {
    Box,
    ButtonBase,
    CircularProgress,
    IconButton,
    InputAdornment,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import type { EnteFile } from "ente-media/file";
import i18next from "i18next";
import React, { useEffect, useRef, useState } from "react";
import {
    findMe,
    searchFinder,
    type FinderConfig,
    type FinderFilter,
    type FindMeError,
} from "./filter";

const copy = {
    en: {
        all: "All photos",
        findMe: "Find me",
        findMeNote:
            "Take or pick a selfie. It is only used for this search and isn't stored.",
        search: "Search: cake, dancing, green dress…",
        you: (n: number) => `Photos of you (${n})`,
        results: (n: number, q: string) => `${n} photos for “${q}”`,
        no_face: "We couldn't see a face. Try a clear selfie in good light.",
        many_faces: "Please use a photo with just you in it.",
        too_large: "That photo is too large. Try a normal selfie.",
        slow_down: "Too many tries. Wait a minute and try again.",
        failed: "Something went wrong. Please try again.",
        nothing: "No photos found.",
        clear: "Clear",
    },
    fr: {
        all: "Toutes les photos",
        findMe: "Me trouver",
        findMeNote:
            "Prenez ou choisissez un selfie. Il sert uniquement à cette recherche et n'est pas conservé.",
        search: "Rechercher : gâteau, danse, robe verte…",
        you: (n: number) => `Vos photos (${n})`,
        results: (n: number, q: string) => `${n} photos pour « ${q} »`,
        no_face:
            "Nous n'avons pas vu de visage. Essayez un selfie net et bien éclairé.",
        many_faces: "Utilisez une photo où vous êtes seul(e).",
        too_large: "Cette photo est trop lourde. Essayez un selfie normal.",
        slow_down: "Trop d'essais. Attendez une minute et réessayez.",
        failed: "Une erreur est survenue. Veuillez réessayer.",
        nothing: "Aucune photo trouvée.",
        clear: "Effacer",
    },
};

const strings = () => {
    const lang = i18next.language || navigator.language || "";
    return lang.toLowerCase().startsWith("fr") ? copy.fr : copy.en;
};

interface FinderBarProps {
    config: FinderConfig;
    files: EnteFile[];
    accessToken: string;
    active: FinderFilter | undefined;
    onChange: (filter: FinderFilter | undefined) => void;
}

/**
 * Folders, "Find me" and search above the grid of a shared album served by
 * our finder service. Each narrows the same grid; one is active at a time.
 */
export const FinderBar: React.FC<FinderBarProps> = ({
    config,
    files,
    accessToken,
    active,
    onChange,
}) => {
    const s = strings();
    const [query, setQuery] = useState("");
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState<string | undefined>();
    const fileInput = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key == "Escape" && active) onChange(undefined);
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [active, onChange]);

    const runFindMe = async (selfie: File) => {
        setBusy(true);
        setMessage(undefined);
        const r = await findMe(accessToken, selfie);
        setBusy(false);
        if ("error" in r) {
            setMessage(s[r.error satisfies FindMeError]);
        } else if (!r.fileIDs.length) {
            setMessage(s.nothing);
        } else {
            onChange({ kind: "find-me", label: "", fileIDs: r.fileIDs });
        }
    };

    const runSearch = async () => {
        const q = query.trim();
        if (!q) return;
        setBusy(true);
        setMessage(undefined);
        const ids = await searchFinder(accessToken, q);
        setBusy(false);
        if (!ids) setMessage(s.failed);
        else if (!ids.length) setMessage(s.nothing);
        else onChange({ kind: "search", label: q, fileIDs: ids });
    };

    const activeLabel =
        active?.kind == "find-me"
            ? s.you(active.fileIDs.length)
            : active?.kind == "search"
              ? s.results(active.fileIDs.length, active.label)
              : undefined;

    return (
        <Stack sx={{ gap: 1.5, px: { xs: "12px", sm: "24px" }, py: 1.5 }}>
            {config.folders.length > 0 && (
                <Stack
                    direction="row"
                    sx={{ gap: 1, overflowX: "auto", pb: 0.5 }}
                >
                    <FolderTile
                        name={s.all}
                        count={files.length}
                        cover={files[0]}
                        selected={!active}
                        onClick={() => onChange(undefined)}
                    />
                    {config.folders.map((folder) => (
                        <FolderTile
                            key={folder.name}
                            name={folder.name}
                            count={folder.fileIDs.length}
                            cover={files.find(
                                (f) => f.id == folder.coverFileID,
                            )}
                            selected={
                                active?.kind == "folder" &&
                                active.name == folder.name
                            }
                            onClick={() =>
                                onChange({
                                    kind: "folder",
                                    name: folder.name,
                                    fileIDs: folder.fileIDs,
                                })
                            }
                        />
                    ))}
                </Stack>
            )}
            <Stack
                direction={{ xs: "column", sm: "row" }}
                sx={{ gap: 1, alignItems: { sm: "center" } }}
            >
                {config.features.findMe && (
                    <>
                        <ButtonBase
                            onClick={() => fileInput.current?.click()}
                            disabled={busy}
                            sx={{
                                gap: 1,
                                px: 2,
                                height: 40,
                                borderRadius: "20px",
                                bgcolor: "accent.main",
                                color: "accent.contrastText",
                                fontWeight: 600,
                                flexShrink: 0,
                            }}
                        >
                            <FaceIcon fontSize="small" />
                            {s.findMe}
                        </ButtonBase>
                        <input
                            ref={fileInput}
                            type="file"
                            accept="image/*"
                            capture="user"
                            hidden
                            onChange={(e) => {
                                const selfie = e.target.files?.[0];
                                e.target.value = "";
                                if (selfie) void runFindMe(selfie);
                            }}
                        />
                    </>
                )}
                {config.features.search && (
                    <TextField
                        size="small"
                        value={query}
                        placeholder={s.search}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key == "Enter") void runSearch();
                        }}
                        sx={{ flex: 1, maxWidth: 480 }}
                        slotProps={{
                            input: {
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <SearchIcon fontSize="small" />
                                    </InputAdornment>
                                ),
                            },
                        }}
                    />
                )}
                {busy && <CircularProgress size={20} />}
            </Stack>
            {config.features.findMe && !activeLabel && !message && (
                <Typography variant="small" sx={{ color: "text.muted" }}>
                    {s.findMeNote}
                </Typography>
            )}
            {message && (
                <Typography variant="small" sx={{ color: "critical.main" }}>
                    {message}
                </Typography>
            )}
            {activeLabel && (
                <Stack direction="row" sx={{ alignItems: "center", gap: 1 }}>
                    <Typography sx={{ fontWeight: 600 }}>
                        {activeLabel}
                    </Typography>
                    <IconButton
                        size="small"
                        aria-label={s.clear}
                        onClick={() => onChange(undefined)}
                    >
                        <CloseIcon fontSize="small" />
                    </IconButton>
                </Stack>
            )}
        </Stack>
    );
};

interface FolderTileProps {
    name: string;
    count: number;
    cover: EnteFile | undefined;
    selected: boolean;
    onClick: () => void;
}

const FolderTile: React.FC<FolderTileProps> = ({
    name,
    count,
    cover,
    selected,
    onClick,
}) => {
    const [url, setUrl] = useState<string | undefined>();
    useEffect(() => {
        let live = true;
        if (cover)
            void downloadManager
                .renderableThumbnailURL(cover)
                .then((u) => live && setUrl(u));
        return () => {
            live = false;
        };
    }, [cover]);

    return (
        <ButtonBase
            onClick={onClick}
            aria-pressed={selected}
            sx={{
                flexShrink: 0,
                width: 148,
                height: 104,
                borderRadius: "12px",
                overflow: "hidden",
                position: "relative",
                outline: selected ? "3px solid" : "none",
                outlineColor: "accent.main",
                outlineOffset: -3,
                bgcolor: "fill.faint",
            }}
        >
            {url && (
                <Box
                    component="img"
                    src={url}
                    alt=""
                    sx={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                    }}
                />
            )}
            <Box
                sx={{
                    position: "absolute",
                    inset: 0,
                    background:
                        "linear-gradient(transparent 40%, rgba(0,0,0,0.65))",
                }}
            />
            <Stack
                sx={{
                    position: "absolute",
                    left: 10,
                    right: 10,
                    bottom: 8,
                    color: "#fff",
                    textAlign: "left",
                }}
            >
                <Typography
                    noWrap
                    sx={{ fontWeight: 600, fontSize: 14, lineHeight: "18px" }}
                >
                    {name}
                </Typography>
                <Typography sx={{ fontSize: 12, opacity: 0.85 }}>
                    {count}
                </Typography>
            </Stack>
        </ButtonBase>
    );
};

import type { CollectionSummary } from "ente-new/photos/services/collection-summary";
import { describe, expect, test } from "vitest";
import {
    albumFolderRows,
    parseAlbumFolderName,
} from "../src/utils/album-folders";

const album = (id: number, name: string, fileCount = 1): CollectionSummary => ({
    id,
    type: "album",
    attributes: new Set(),
    name,
    fileCount,
    latestFile: undefined,
    coverFile: undefined,
    updationTime: undefined,
    sortPriority: 0,
});

describe("album folder names", () => {
    test("splits on a middle dot or a slash", () => {
        expect(parseAlbumFolderName("Wedding · Repas")).toEqual({
            folder: "Wedding",
            child: "Repas",
        });
        expect(parseAlbumFolderName("Wedding / Mairie")).toEqual({
            folder: "Wedding",
            child: "Mairie",
        });
        expect(parseAlbumFolderName("Wedding · Couple : Groupes")).toEqual({
            folder: "Wedding",
            child: "Couple : Groupes",
        });
    });

    test("leaves plain names and names with an empty side alone", () => {
        expect(parseAlbumFolderName("Couple : Groupes")).toBeUndefined();
        expect(parseAlbumFolderName("AC/DC")).toBeUndefined();
        expect(parseAlbumFolderName(" · Repas")).toBeUndefined();
        expect(parseAlbumFolderName("Wedding · ")).toBeUndefined();
    });
});

describe("album folder rows", () => {
    const albums = [
        album(1, "Favorites"),
        album(2, "Wedding · Mairie"),
        album(3, "Trip"),
        album(4, "Wedding · Repas"),
        album(5, "wedding", 0),
    ];

    test("collapsed folder sits where its first album was", () => {
        const rows = albumFolderRows(albums, new Set());
        expect(rows.map((r) => r.type)).toEqual([
            "collection",
            "folder",
            "collection",
        ]);
        const folder = rows[1]!;
        expect(folder.type == "folder" && folder.name).toBe("Wedding");
        expect(folder.type == "folder" && folder.albumCount).toBe(2);
    });

    test("an album named like the folder becomes the folder itself", () => {
        const rows = albumFolderRows(albums, new Set());
        const folder = rows[1]!;
        expect(folder.type == "folder" && folder.collectionSummary?.id).toBe(5);
        expect(
            rows.some(
                (r) => r.type == "collection" && r.collectionSummary.id == 5,
            ),
        ).toBe(false);
    });

    test("an expanded folder lists its albums under it with short names", () => {
        const rows = albumFolderRows(albums, new Set(["wedding"]));
        expect(
            rows.map((r) =>
                r.type == "folder"
                    ? `folder:${r.name}`
                    : `${r.depth}:${r.label}`,
            ),
        ).toEqual([
            "0:Favorites",
            "folder:Wedding",
            "1:Mairie",
            "1:Repas",
            "0:Trip",
        ]);
    });

    test("folder names match case-insensitively", () => {
        const rows = albumFolderRows(
            [album(1, "Wedding · A"), album(2, "wedding · B")],
            new Set(),
        );
        expect(rows).toHaveLength(1);
    });

    test("a single same-named album without children stays a plain album", () => {
        const rows = albumFolderRows([album(1, "Wedding")], new Set());
        expect(rows).toHaveLength(1);
        expect(rows[0]).toMatchObject({
            type: "collection",
            label: "Wedding",
            depth: 0,
        });
    });
});

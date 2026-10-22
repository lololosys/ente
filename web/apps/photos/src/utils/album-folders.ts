import type { CollectionSummary } from "ente-new/photos/services/collection-summary";

/**
 * Album folders exist only in this web app: an album named "Wedding · Repas"
 * (or "Wedding / Repas") is shown as "Repas" inside a "Wedding" folder in the
 * sidebar. Nothing is stored; the name is the whole contract, so the mobile
 * apps and other clients keep working and just show the full name.
 *
 * An album named exactly like the folder ("Wedding") becomes the folder row
 * itself, so the folder can also be opened as an album.
 */
export const parseAlbumFolderName = (
    name: string,
): { folder: string; child: string } | undefined => {
    const match = /^(.*?\S)\s+[·/]\s+(\S.*)$/.exec(name);
    return match ? { folder: match[1]!, child: match[2]! } : undefined;
};

export type AlbumFolderRow =
    | {
          type: "collection";
          collectionSummary: CollectionSummary;
          /** The name to show: the part after the folder for nested albums. */
          label: string;
          depth: 0 | 1;
      }
    | {
          type: "folder";
          /** Lowercased folder name, the key for expanded state. */
          key: string;
          name: string;
          albumCount: number;
          /** The album named exactly like the folder, if any. */
          collectionSummary: CollectionSummary | undefined;
          expanded: boolean;
      };

export const albumFolderKey = (name: string) => name.trim().toLowerCase();

/**
 * Turns an already sorted list of albums into sidebar rows. A folder takes
 * the place of its first album in the sort order; its albums keep their
 * relative order inside it.
 */
export const albumFolderRows = (
    collectionSummaries: CollectionSummary[],
    expandedFolders: ReadonlySet<string>,
): AlbumFolderRow[] => {
    const children = new Map<
        string,
        { name: string; items: { cs: CollectionSummary; child: string }[] }
    >();
    for (const cs of collectionSummaries) {
        const parsed = parseAlbumFolderName(cs.name);
        if (!parsed) continue;
        const key = albumFolderKey(parsed.folder);
        const entry = children.get(key) ?? { name: parsed.folder, items: [] };
        entry.items.push({ cs, child: parsed.child });
        children.set(key, entry);
    }

    const heads = new Map<string, CollectionSummary>();
    for (const cs of collectionSummaries) {
        const key = albumFolderKey(cs.name);
        if (children.has(key) && !heads.has(key)) heads.set(key, cs);
    }

    const rows: AlbumFolderRow[] = [];
    const placed = new Set<string>();
    for (const cs of collectionSummaries) {
        const parsed = parseAlbumFolderName(cs.name);
        const key = parsed
            ? albumFolderKey(parsed.folder)
            : heads.get(albumFolderKey(cs.name)) === cs
              ? albumFolderKey(cs.name)
              : undefined;
        if (key === undefined) {
            rows.push({
                type: "collection",
                collectionSummary: cs,
                label: cs.name,
                depth: 0,
            });
            continue;
        }
        if (placed.has(key)) continue;
        placed.add(key);
        const { name, items } = children.get(key)!;
        const expanded = expandedFolders.has(key);
        rows.push({
            type: "folder",
            key,
            name,
            albumCount: items.length,
            collectionSummary: heads.get(key),
            expanded,
        });
        if (expanded) {
            for (const { cs: child, child: label } of items) {
                rows.push({
                    type: "collection",
                    collectionSummary: child,
                    label,
                    depth: 1,
                });
            }
        }
    }
    return rows;
};

/** The folder that holds an album, so the sidebar can open it when needed. */
export const albumFolderKeyOf = (
    collectionSummaries: CollectionSummary[],
    collectionID: number | undefined,
): string | undefined => {
    const cs = collectionSummaries.find(({ id }) => id == collectionID);
    const parsed = cs && parseAlbumFolderName(cs.name);
    return parsed ? albumFolderKey(parsed.folder) : undefined;
};

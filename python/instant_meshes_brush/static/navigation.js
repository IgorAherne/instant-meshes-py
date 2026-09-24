/*
    navigation.js: which camera move a drag makes.

    One table for every mouse button and modifier, kept free of three.js so it
    can be read -- and tested -- on its own.  It is the navigation people bring
    from the programs they already use: Blender's middle button, 3ds Max's
    Alt + middle, Maya's and Unity's Alt + left and Alt + right.  A host page
    that shows models in its own viewer can copy it, and the two then feel the
    same in the hand (Spellcast3D's model viewer does exactly that).

        wheel                  zoom toward the cursor   (OrbitControls, not a drag)
        middle drag            orbit                    (Blender)
        Alt + middle drag      orbit                    (3ds Max)
        Shift + middle drag    pan                      (Blender)
        Ctrl + middle drag     zoom                     (Blender)
        right drag             pan
        Alt + right drag       zoom                     (Maya, Unity)
        left drag              whatever the viewer gives it: the brush here,
                               orbit in a viewer that has no brush
        Alt + left drag        orbit                    (Maya, Unity)
        F                      frame the model          (tools.js)
*/

export const ORBIT = 'orbit';
export const PAN = 'pan';
export const ZOOM = 'zoom';

/**
 * The camera move a drag starts, or null when the drag is not the camera's.
 *
 * Ctrl wins over Shift on the middle button, and Meta counts as Ctrl, so a
 * Mac's Cmd + middle zooms as Ctrl + middle does elsewhere.
 *
 * @param {number} button  PointerEvent.button: 0 left, 1 middle, 2 right
 * @param {{altKey?: boolean, shiftKey?: boolean, ctrlKey?: boolean,
 *          metaKey?: boolean}} keys  the modifiers held when it started
 * @param {string|null} left  what a plain left drag does: ORBIT, or null
 *        where the left button belongs to something else
 * @returns {'orbit'|'pan'|'zoom'|null}
 */
export function dragAction(button, keys, left) {
    const ctrl = Boolean(keys.ctrlKey || keys.metaKey);
    switch (button) {
        case 0:
            return keys.altKey ? ORBIT : left;
        case 1:
            if (ctrl) return ZOOM;
            return keys.shiftKey ? PAN : ORBIT;
        case 2:
            return keys.altKey ? ZOOM : PAN;
        default:
            return null;
    }
}

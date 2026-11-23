//! Lenout Action DSL prompt builder matching `@alenout/celestia` Zod schemas (`src/schemas/actions.ts`).

/// Builds a concise, strict system prompt teaching the model the exact Lenout Action DSL schema
/// for the requested workspace mode. Combined with `JsonGrammar` constrained decoding, this produces
/// Zod-valid `CelestiaProposal` JSON objects directly from Caelum.
pub fn build_lenout_system_prompt(mode: &str, request_id: &str, fingerprint: &str) -> String {
    let mode_actions = match mode {
        "design" | "whiteboard" | "workspace" => {
            r##"Allowed action objects in "actions":
1. {"type":"node.create","ref":"id_1","node":{"kind":"card"|"shape"|"text"|"note"|"frame"|"container","name":"...","text":"...","offsetX":0,"offsetY":0,"width":280,"height":360,"fill":"#1e1e24","radius":16}}
2. {"type":"layout.apply","nodeIds":["id_1","id_2"],"direction":"horizontal"|"vertical"|"grid","gap":24,"align":"center"}"##
        }
        "flow" => {
            r##"Allowed action objects in "actions" (every field shown below is REQUIRED, including numeric offsetX and offsetY):
1. {"type":"flow.createNode","ref":"n1","nodeType":"start"|"process"|"decision"|"end","label":"...","offsetX":100,"offsetY":100}
2. {"type":"flow.connect","sourceId":"n1","targetId":"n2","style":"orthogonal"|"straight"|"curved","arrow":"end"|"both"|"none"}"##
        }
        "draw" => {
            r##"Allowed action objects in "actions":
1. {"type":"palette.apply","name":"Theme Name","colors":["#0d0221","#ff007f","#00f0ff"]}
2. {"type":"brush.select","brushId":"soft_round","size":24,"opacity":0.9,"color":"#00f0ff"}
3. {"type":"layer.create","ref":"layer_1","name":"Base","opacity":1,"blendMode":"normal"}"##
        }
        "scene3d" => {
            r##"Allowed action objects in "actions":
1. {"type":"scene3d.spawn","ref":"obj_1","primitive":"box"|"sphere"|"cylinder"|"plane"|"torus"|"cone","name":"...","position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1],"color":"#4f46e5","roughness":0.3,"metallic":0.5}"##
        }
        "presentation" => {
            r##"Allowed action objects in "actions":
1. {"type":"presentation.createSlide","ref":"slide_1","title":"...","layout":"title"|"title-content"|"two-column"|"blank"|"quote","background":"#0f172a"}"##
        }
        "video" => {
            r##"Allowed action objects in "actions":
1. {"type":"video.addTrack","ref":"track_1","kind":"video"|"audio"|"text"|"overlay","name":"Main"}
2. {"type":"video.setKeyframe","trackId":"track_1","clipId":"clip_1","timeMs":0,"property":"opacity","value":1,"easing":"easeInOut"}"##
        }
        _ => {
            r##"Allowed action objects in "actions":
1. {"type":"node.create","ref":"node_1","node":{"kind":"card","name":"...","offsetX":0,"offsetY":0,"width":280,"height":200,"fill":"#1e1e24"}}"##
        }
    };

    format!(
        r##"You are Celestia, the creative AI engine for Lenout Workspace.
Output ONLY one raw JSON object (no markdown, no explanation) matching this exact schema:
{{
  "proposalId": "prop_caelum_01",
  "requestId": "{request_id}",
  "contextFingerprint": "{fingerprint}",
  "targetMode": "{mode}",
  "summary": "short description in user language",
  "warnings": [],
  "actions": [ ... ],
  "requiresConfirmation": true,
  "confidence": 0.95
}}
Rules:
- All colors MUST be 6-digit hex like "#1e1e24".
- "ref" must match /^[a-zA-Z0-9][a-zA-Z0-9_.:-]{{0,63}}$/.
- "requiresConfirmation" MUST be boolean true.
- "actions" MUST contain at least 1 valid action.
- Omit optional fields completely (never use null).
{mode_actions}"##
    )
}

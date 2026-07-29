/**
 * The user simulators the leaderboard treats as its reference.
 *
 * Voice uses the versioned v1.x lineage (git tag `voice-user-sim-<version>`);
 * text has used gpt-4.1 and gpt-5.2 across eras.
 *
 * Running a different simulator does not make a submission custom — the scaffold
 * is unchanged — but a stronger simulator measurably shifts scores, so those rows
 * are flagged rather than silently ranked alongside reference runs.
 */
const REFERENCE_TEXT_USER_SIMS = ['gpt-4.1', 'gpt-5.2']

export const isReferenceUserSim = (userSimulator, isVoice) => {
  if (!userSimulator) return true
  return isVoice
    ? /^v\d/.test(userSimulator)
    : REFERENCE_TEXT_USER_SIMS.some(prefix => userSimulator.startsWith(prefix))
}

export const NON_REFERENCE_USER_SIM_TITLE =
  'Non-reference user simulator — scores are not directly comparable to rows run with the reference simulator'

/** Suffix for compact listings (dropdowns), empty when the simulator is the reference one. */
export const nonReferenceUserSimLabel = (userSimulator, isVoice) =>
  isReferenceUserSim(userSimulator, isVoice) ? '' : ` — user sim: ${userSimulator}`

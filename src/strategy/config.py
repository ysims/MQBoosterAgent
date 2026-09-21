"""Strategy tuning constants: guarding, support, kickoff, and positioning."""

from __future__ import annotations


# ======================================================================
# Kick power
# ======================================================================

KICK_POWER_OUR_KICKOFF = 5.0

# ======================================================================
# Player skill parameters: guarding and support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7

SUPPORT_DIST_M = 3.0

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevent attacker-selection oscillation
GUARD_KEEP_DIST_MARGIN_M = 0.3     # Prevent guard-selection oscillation

# A player's own ball distance jumps to infinity the instant its own vision
# loses the ball (see Context.ball's docstring -- there's no shared/fused
# reading to fall back on), which would otherwise fail the keep-margin check
# above immediately and hand the attacker role to whichever other player
# currently has any ball reading at all, even briefly. This cooldown holds
# the current attacker for a short window after any switch regardless of
# ball distance, giving Player._search_for_ball() (see strategy/player.py)
# a real chance to reacquire the ball before role reassignment reconsiders.
# Kept longer than planning.config.SEARCH_MEMORY_SEC on purpose: a shorter
# cooldown would let the attacker get reassigned mid-search while its
# remembered ball position is still fresh, cutting the search off before it
# had the full window it was supposed to get.
ATTACKER_SWITCH_COOLDOWN_SEC = 6.0

FALLEN_COST = 10.0  # Distance penalty assigned to a fallen player (m)

# ======================================================================
# Kickoff and set-play strategy
# ======================================================================

# Kickoff
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

CENTER_LEAVE_DIST_M = 0.15 # Distance from center at which the ball counts as moved

OPP_SET_WALL_DIST_M = 2.0 # Blocking distance during an opponent kickoff

# ======================================================================
# Positioning and avoidance
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.65
CIRCLE_MARGIN_M = 0.3

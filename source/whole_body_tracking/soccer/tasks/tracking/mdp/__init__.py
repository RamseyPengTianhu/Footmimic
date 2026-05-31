"""This sub-module contains the functions that are specific to the locomotion environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from soccer.tasks.tracking.mdp import *  # noqa: F401, F403

# from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .rewards_dribbling import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

from .commands_multi_motion_soccer import *  # noqa: F401, F403
# from .commands import *

# v10 event-conditioned kick modules
from . import event_phase  # noqa: F401
from . import observations_v10 as obs_v10  # noqa: F401
from . import rewards_v10  # noqa: F401
from . import rewards_v35  # noqa: F401
from . import rewards_v36  # noqa: F401
from . import rewards_v36b  # noqa: F401
from . import terminations_v36  # noqa: F401
from . import phase_tracking_v36  # noqa: F401
from . import phase_tracking_v36b  # noqa: F401
from . import phase_vae_prior  # noqa: F401
from . import content_cvae_prior  # noqa: F401
from . import v3_cvae_rewards  # noqa: F401
from . import latent_prior_command  # noqa: F401
from . import latent_prior_rewards  # noqa: F401
from . import latent_prior_observations  # noqa: F401
from .event_conditioned_obs_builder import V10ObsBuilder  # noqa: F401
from . import rewards_student  # noqa: F401

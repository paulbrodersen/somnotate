#!/usr/bin/env python
"""
Simple tool for manual state annotation.

TODO:
- automatically add user defined keybindings to the help string
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
from matplotlib.collections import LineCollection

DEBUG = False
STATE_LINE_WIDTH = 5
EPS = 1e-9


class TimeSeriesStateViewer(object):
    """
    Simple GUI to annotate time series data with non-overlapping state intervals.

    Arguments:
    ----------
    data_axis : matplotlib axis instance
        Axis displaying the time series data.
    state_axis : matplotlib axis instance
        Axis used to display the state annotations.
    state_to_color : dict state id : matplotlib color argument (optional)
        Colors to use for each state in the state annotation plot.
    state_display_order : list of state ids (optional)
        Order of states on the y-axis in state annotation plot.
    default_selection_length : int/float (optional, default 4)
        Default x interval length for a selection.
    default_view_length : int/float (optional, default 60)
        Default x-limit width (i.e. number of time points displayed at any point in time).
    interval_to_state : dict (float start, float stop) : state id (optional, default None)
        Predefined state annotation.
    regions_of_interest: list of (float start, float stop) tuples
        Predefined regions of interest. Press 'h' to see how to quickly navigate these regions.
    selection_callback: function or None (default None)
        Function that is called upon each new selection with selection lower bound
        and selection upper bound as arguments.

    Example:
    --------
    ```
    import numpy as np
    import matplotlib.pyplot as plt
    from _manual_state_annotation import TimeSeriesAnnotator

    # initialise annotator object
    fig, (data_axis, state_axis) = plt.subplots(2, 1)
    data_axis.plot(np.random.rand(1000))
    keymap = {'a' : 'state A', 'b' : 'state B'}
    annotator = TimeSeriesAnnotator(data_axis, state_axis, keymap)
    plt.show()

    # annotate states by pressing 'a' or 'b'

    # retrieve annotation
    annotation = annotator.interval_to_state
    ```

    Notes:
    ------
    Press 'h' to display the interactive help.
    A summary of all navigation and selection commands can be found there.
    """

    def __init__(self, data_axis, state_axis,
                 state_to_color                 = None,
                 state_display_order            = None,
                 default_selection_length       = 4,
                 default_view_length            = 60,
                 interval_to_state              = None,
                 regions_of_interest            = None,
                 selection_callback             = None,
    ):
        # define navigation keys
        self.basic_movement_keys = [
            'right',
            'left',
            'ctrl+right',
            'ctrl+left',
            'end',
            'home',
            'alt+right',
            'alt+left',
        ]
        self.roi_movement_keys = [
            'down',
            'up',
            'ctrl+down',
            'ctrl+up',
            'pagedown',
            'pageup',
        ]
        self.interval_movement_keys = [
            '[',
            ']',
            'ctrl+[',
            'ctrl+]',
            'alt+[',
            'alt+]',
        ]
        self.goto_navigation_keys = [str(ii) for ii in range(10)] + ['enter', 'backspace', '.', '-']

        self.movement_keys = self.basic_movement_keys \
                             + self.roi_movement_keys \
                             + self.interval_movement_keys \
                             + self.goto_navigation_keys

        # bookkeeping
        self.data_axis                = data_axis
        self.state_axis               = state_axis
        self.default_selection_length = default_selection_length
        self.default_view_length      = default_view_length
        self.rois                     = regions_of_interest
        self._selection_callback      = selection_callback

        # initialize state annotation and plot
        self._initialize_state_annotation(
            interval_to_state   = interval_to_state,
            state_to_color      = state_to_color,
            state_display_order = state_display_order
        )

        # initialize ROIs
        if not (self.rois is None):
            self.current_roi_index = -1
            self.total_rois = len(self.rois)

        # initialise view
        self.figure = self.data_axis.get_figure()
        self.data_min, self.data_max = self.data_axis.dataLim.intervalx
        self._initialize_selection(self.data_min, self.data_min+default_selection_length)
        self._set_view(self.data_min, self.data_min+default_view_length)

        # initialise callbacks
        self.figure.canvas.mpl_connect('button_press_event'  , self._on_button_press)
        self.figure.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.figure.canvas.mpl_connect('axes_leave_event'    , self._on_axes_leave)
        self.figure.canvas.mpl_connect('key_press_event'     , self._on_key_press)
        #  self.figure.canvas.mpl_connect('key_release_event'   , self._on_key_release)
        self.figure.canvas.mpl_connect('motion_notify_event' , self._on_motion)
        self.button_press_start = None

        self.memory = ''


    def _check_keybindings(self, key_to_state):

        def check(reserved_keys, purpose):
            conflicting_keys = []
            for key in key_to_state.keys():
                if key in reserved_keys:
                    conflicting_keys.append(key)

            if len(conflicting_keys) != 0:
                error_message = "\n\nThe following keys are reserved {} and cannot be used for state annotation:\n".format(purpose)
                for key in conflicting_keys:
                    error_message += '\t{}\n'.format(key)
                error_message += "\n"
                error_message += "Please change your bindings in `key_to_state`.\n"
                error_message += "Keys reserved {} include:\n".format(purpose)
                for key in reserved_keys:
                    error_message += '\t{}\n'.format(key)
                raise ValueError(error_message)

        # check(['l', 'L', 'q', 's', 'g', 'G', 'k'], 'by matplotlib')
        check(self.movement_keys, 'for navigation')


    def _display_help(self):
        help_string = """
        Mouse behaviour:
        ================

        On data axis:
        -------------
        left click         -- select epoch
        hold left and drag -- make arbitrary selection
        shift + left click -- expand selection to point

        On state axis:
        --------------
        hold left click on state transition marker -- move state transition


        Keybindings:
        ============

        General
        -------
        ?         -- display this help

        Basic navigation:
        -----------------
        left       -- move backward by one epoch length (`default_selection_length`)
        right      -- move forward  by one epoch length
        ctrl+left  -- move to the preceding view (fast scroll backward)
        ctrl+right -- move to the following view (fast scroll forward)
        Home       -- move to the start of the time series
        End        -- move to the end of the time series
        alt+left   -- expand selection backward by one epoch length
        alt+right  -- expand selection forward  by one epoch length

        Regions of interest (ROI) navigation:
        -------------------------------------
        up         -- move to the previous ROI
        down       -- move to the next ROI
        ctrl+up    -- move backward in the list of ROIs by 10% (fast scroll backward)
        ctrl+down  -- move forward  in the list of ROIs by 10% (fast scroll forward)
        PageUp     -- move to first ROI
        PageDown   -- move to last ROI

        State interval navigation:
        --------------------------
        [          -- move to the start of the current state interval (or of the preceding state if already at the start)
        ]          -- move to the end   of the current state interval (or of the following state if already at the end)
        ctrl+[     -- move to the end   of the preceding interval with the same state as the current state
        ctrl+]     -- move to the start of the following interval with the same state as the current state
        alt+[      -- expand selection to the start of the current state interval (or of the preceding state if already at the start)
        alt+]      -- expand selection to the end   of the current state interval (or of the following state if already at the end)

        GOTO navigaton:
        ---------------
        Enter valid float and press enter; backspace to clear memory.

        Contact:
        ========
        Please raise any issues you encounter at:
        www.github.com/paulbrodersen/time_series_annotator/issues

        """
        print(help_string)


    def _on_button_press(self, event):
        if event.inaxes is self.data_axis:
            self.button_press_start = event.xdata


    def _on_button_release(self, event):

        if event.inaxes is self.data_axis:
            if self.button_press_start:
                if np.abs(self.button_press_start - event.xdata) / self.default_selection_length < 0.01: # i.e. hardly any motion; presumably just a short click
                    self._handle_click(event)
                self.button_press_start = None


    def _handle_click(self, event):

        if event.key is None: # just select an epoch
            epoch_lower_bound = int(event.xdata/self.default_selection_length) * self.default_selection_length
            epoch_upper_bound = epoch_lower_bound + self.default_selection_length
            self._update_selection(epoch_lower_bound, epoch_upper_bound)

        elif event.key == 'shift': # expand current selection to point
            self._update_selection(
                np.min([self.selection_lower_bound, event.xdata]),
                np.max([self.selection_upper_bound, event.xdata]))


    def _handle_hold_click(self, event):
        self._update_selection(*sorted([self.button_press_start, event.xdata]))


    def _on_axes_leave(self, event):
        self._on_button_release(event)


    def _on_motion(self, event):
        if event.inaxes is self.data_axis:
            if self.button_press_start:
                self._handle_hold_click(event)


    def _on_key_press(self, event):
        if DEBUG:
            print(event.key)

        if event.key in self.basic_movement_keys:
            self._basic_navigation(event)

        elif event.key in self.roi_movement_keys:
            if self.rois is None:
                print("Warning: an ROI movement/selection key was pressed but there are no regions of interest defined!")
            else:
                self._roi_navigation(event)

        elif event.key in self.interval_movement_keys:
            if self.interval_to_state is None:
                print("Warning: an state interval movement/selection key was pressed but there are no state intervals defined!")
            else:
                self._state_interval_navigation(event)

        elif event.key in self.goto_navigation_keys:
            self._goto_navigation(event)

        elif event.key == '?':
            self._display_help()


    def _basic_navigation(self, event):
        if event.key == 'right':
            self._move_to_next_epoch()

        elif event.key == 'ctrl+right':
            self._move_to_next_view()

        elif event.key == 'left':
            self._move_to_previous_epoch()

        elif event.key == 'ctrl+left':
            self._move_to_previous_view()

        elif event.key == 'end':
            self._move_to_last_view()

        elif event.key == 'home':
            self._move_to_first_view()

        elif event.key == 'alt+left':
            self._expand_selection_to_previous_epoch()

        elif event.key == 'alt+right':
            self._expand_selection_to_next_epoch()

        # elif event.key == 'c':
        #     self._center_view_on_selection()


    def _roi_navigation(self, event):
        if event.key == 'down':
            self._select_next_roi()

        elif event.key == 'ctrl+down':
            self._jump_several_rois_forward()

        elif event.key == 'pagedown':
            self._select_last_roi()

        elif event.key == 'up':
            self._select_previous_roi()

        elif event.key == 'ctrl+up':
            self._jump_several_rois_backward()

        elif event.key == 'pageup':
            self._select_first_roi()


    def _state_interval_navigation(self, event):
        if event.key == '[':
            self._move_to_interval_start()

        elif event.key == ']':
            self._move_to_interval_stop()

        elif event.key == 'ctrl+[':
            self._move_to_previous_interval_with_same_state()

        elif event.key == 'ctrl+]':
            self._move_to_next_interval_with_same_state()

        elif event.key == 'alt+[':
            self._select_to_interval_start()

        elif event.key =='alt+]':
            self._select_to_interval_stop()


    def _goto_navigation(self, event):
        if event.key in '0123456789.-':
            self.memory += event.key

        elif event.key == 'enter':
            if len(self.memory) > 0:
                x = float(self.memory)
                self._update_selection(x, x+self.default_selection_length)
                self._center_view_on_selection()
                self.memory = ''

        elif event.key == 'backspace':
            self.memory = ''


    def _move_to_next_epoch(self):
        self._update_selection(self.selection_upper_bound,
                               self.selection_upper_bound + self.default_selection_length)
        # if necessary, jump to next view
        if self.selection_lower_bound >= self.view_upper_bound: # NOTE: this is the old selection_upper_bound!
            self._set_view(self.view_upper_bound,
                              self.view_upper_bound + self.default_view_length)


    def _move_to_previous_epoch(self):
        self._update_selection(self.selection_lower_bound - self.default_selection_length,
                               self.selection_lower_bound)
        # if necessary, jump to previous view
        if self.selection_upper_bound <= self.view_lower_bound: # NOTE: this is the old selection_lower_bound!
            self._set_view(self.view_lower_bound - self.default_view_length,
                              self.view_lower_bound)


    def _expand_selection_to_next_epoch(self):
        self._update_selection(self.selection_lower_bound,
                               self.selection_upper_bound + self.default_selection_length)
        if self.selection_upper_bound > self.view_upper_bound:
            self._set_view(self.view_upper_bound,
                              self.view_upper_bound + self.default_view_length)


    def _expand_selection_to_previous_epoch(self):
        self._update_selection(self.selection_lower_bound - self.default_selection_length,
                               self.selection_upper_bound)
        if self.selection_lower_bound < self.view_lower_bound:
            self._set_view(self.view_lower_bound - self.default_view_length,
                              self.view_lower_bound)


    def _move_to_next_view(self):
        self._update_selection(self.view_upper_bound,
                               self.view_upper_bound + self.default_selection_length)
        self._set_view(self.view_upper_bound,
                            self.view_upper_bound + self.default_view_length)


    def _move_to_previous_view(self):
        self._update_selection(self.view_lower_bound - self.default_selection_length,
                               self.view_lower_bound)
        self._set_view(self.view_lower_bound - self.default_view_length,
                            self.view_lower_bound)


    def _move_to_last_view(self):
        self._update_selection(self.data_max - self.default_selection_length, self.data_max)
        self._set_view(self.data_max - self.default_view_length, self.data_max)


    def _move_to_first_view(self):
        self._update_selection(self.data_min, self.data_min + self.default_selection_length)
        self._set_view(self.data_min, self.data_min + self.default_view_length)


    def _select_next_roi(self):
        if self.current_roi_index < self.total_rois-1:
            self.current_roi_index += 1
        self._select_roi(*self.rois[self.current_roi_index])


    def _select_previous_roi(self):
        if self.current_roi_index > 0:
            self.current_roi_index -= 1
        self._select_roi(*self.rois[self.current_roi_index])


    def _jump_several_rois_forward(self):
        self.current_roi_index = int(np.min([
            self.current_roi_index + np.max([0.1 * self.total_rois, 1]),
            self.total_rois -1
        ]))
        self._select_roi(*self.rois[self.current_roi_index])


    def _jump_several_rois_backward(self):
        self.current_roi_index = int(np.max([
            self.current_roi_index - np.max([0.1 * self.total_rois, 1]),
            0,
        ]))
        self._select_roi(*self.rois[self.current_roi_index])


    def _select_first_roi(self):
        self.current_roi_index = 0
        self._select_roi(*self.rois[self.current_roi_index])


    def _select_last_roi(self):
        self.current_roi_index = self.total_rois -1
        self._select_roi(*self.rois[self.current_roi_index])


    def _select_roi(self, start, stop):
        self._update_selection(start, stop)

        # If the ROI fits within the default view length, simply center on selection.
        # Otherwise, expand the view to show everything.
        delta = stop - start
        if delta < self.default_view_length:
            self._center_view_on_selection()
        else:
            padding = 0.1 * delta
            self._set_view(start-padding, stop+padding)


    def _move_to_interval_start(self):
        interval = self._get_interval_at(self.selection_lower_bound -EPS)
        if interval:
            start, stop = interval
            self._update_selection(start, start + self.default_selection_length)
            if self.selection_lower_bound < self.view_lower_bound:
                # self._set_view(start, start + self.default_view_length)
                self._center_view_on_selection()
        else:
            print('No interval to go to!')


    def _move_to_interval_stop(self):
        interval = self._get_interval_at(self.selection_upper_bound +EPS)
        if interval:
            start, stop = interval
            self._update_selection(stop - self.default_selection_length, stop)
            if self.selection_upper_bound > self.view_upper_bound:
                # self._set_view(stop - self.default_view_length, stop)
                self._center_view_on_selection()
        else:
            print('No interval to go to!')


    def _select_to_interval_start(self):
        interval = self._get_interval_at(self.selection_lower_bound -EPS)
        if interval:
            start, stop = interval
            self._update_selection(start, self.selection_upper_bound)
            if self.selection_lower_bound < self.view_lower_bound:
                # self._set_view(start, start + self.default_view_length)
                self._center_view_on(start)
        else:
            print('No interval to go to!')


    def _select_to_interval_stop(self):
        interval = self._get_interval_at(self.selection_upper_bound +EPS)
        if interval:
            start, stop = interval
            self._update_selection(self.selection_lower_bound, stop)
            if self.selection_upper_bound > self.view_upper_bound:
                # self._set_view(stop - self.default_view_length, stop)
                self._center_view_on(stop)
        else:
            print('No interval to go to!')


    def _move_to_previous_interval_with_same_state(self):
        current_start, current_stop = self._get_interval_at(self.selection_lower_bound +EPS)
        current_state = self.interval_to_state[(current_start, current_stop)]
        intervals = [(start, stop) for (start, stop), state in self.interval_to_state.items() if (state == current_state) and (stop < current_start)]
        if intervals:
            start, stop = max(intervals, key=lambda x:x[1])
            selection_length = min([stop-start, self.default_selection_length])
            self._update_selection(stop - selection_length, stop)
            if self.selection_lower_bound < self.view_lower_bound:
                # self._set_view(stop - self.default_view_length, stop)
                self._center_view_on_selection()
        else:
            print("No interval to go to!")


    def _move_to_next_interval_with_same_state(self):
        current_start, current_stop = self._get_interval_at(self.selection_upper_bound -EPS)
        current_state = self.interval_to_state[(current_start, current_stop)]
        intervals = [(start, stop) for (start, stop), state in self.interval_to_state.items() if (state == current_state) and (start > current_stop)]
        if intervals:
            start, stop = min(intervals, key=lambda x:x[0])
            selection_length = min([stop-start, self.default_selection_length])
            self._update_selection(start, start + selection_length)
            if self.selection_upper_bound > self.view_upper_bound:
                # self._set_view(start, start + self.default_view_length)
                self._center_view_on_selection()
        else:
            print("No interval to go to!")


    def _set_view(self, view_lower_bound, view_upper_bound):
        self.view_lower_bound = view_lower_bound
        self.view_upper_bound = view_upper_bound
        self.data_axis.set_xlim(self.view_lower_bound, self.view_upper_bound)
        self.state_axis.set_xlim(self.view_lower_bound, self.view_upper_bound)
        self.figure.canvas.draw_idle()


    def _initialize_selection(self, selection_lower_bound, selection_upper_bound):
        self.selection_lower_bound = selection_lower_bound
        self.selection_upper_bound = selection_upper_bound
        self.rect = self.data_axis.axvspan(self.selection_lower_bound, self.selection_upper_bound,
                                    color  = 'whitesmoke',
                                    zorder = -1)


    def _update_selection(self, selection_lower_bound, selection_upper_bound):
        vertices = self.rect.get_xy()
        # vertices are returned in the following order
        #     - lower-left
        #     - upper-left
        #     - upper-right
        #     - lower-right
        #     - lower-left

        # set new x-values for vertices
        vertices[[0, 1, -1], 0] = selection_lower_bound
        vertices[[2, 3],     0] = selection_upper_bound

        # update rectangle coordinates and epoch bounds
        self.rect.set_xy(vertices)
        self.selection_lower_bound = selection_lower_bound
        self.selection_upper_bound = selection_upper_bound
        self.figure.canvas.draw_idle()

        if self._selection_callback:
            self._selection_callback(self.selection_lower_bound, self.selection_upper_bound)


    def _center_view_on_selection(self):
        midpoint = self.selection_lower_bound + 0.5 * (self.selection_upper_bound - self.selection_lower_bound)
        self._center_view_on(midpoint)


    def _center_view_on(self, x):
        self._set_view(x - 0.5 * self.default_view_length,
                       x + 0.5 * self.default_view_length)


    def _initialize_state_annotation(self, interval_to_state,
                                     state_to_color      = None,
                                     state_display_order = None,
    ):
        if interval_to_state is None:
            self.interval_to_state = dict()
        else:
            self.interval_to_state = dict(interval_to_state)

        if state_to_color is None:
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            self.state_to_color = dict()
            for ii, state in enumerate(key_to_state.values()):
                self.state_to_color[state] = color_cycle[ii % len(color_cycle)]
        else:
            self.state_to_color = state_to_color

        if state_display_order is None:
            self.state_display_order = list(self.key_to_state.values())
        else:
            self.state_display_order = state_display_order

        # plot lines
        self.state_to_yvalue = {state : ii for ii, state in enumerate(self.state_display_order[::-1])}
        self.line_artists = dict()
        for (start, stop), state in self.interval_to_state.items():
            self.line_artists[(start, stop)], = self.state_axis.plot(
                (start, stop), (self.state_to_yvalue[state], self.state_to_yvalue[state]),
                color     = self.state_to_color[state],
                linewidth = STATE_LINE_WIDTH,
            )

        # label states on the y-axis
        yticklabels, yticks = zip(*self.state_to_yvalue.items())
        self.state_axis.set_yticks(yticks)
        self.state_axis.set_yticklabels(yticklabels)


    def _get_interval_at(self, x):
        intervals = np.array(list(self.interval_to_state.keys()))
        is_within = np.logical_and(x >= intervals[:, 0], x < intervals[:, 1])
        if np.any(is_within):
            (start, stop), = intervals[is_within]
            return (start, stop)
        else:
            return None


def _disable_matplotlib_keybindings(keep=[]):
    for k, v in list(plt.rcParams.items()):
        if ('keymap.' in k) and not (k in keep):
            plt.rcParams[k] = ""


class TimeSeriesStateAnnotator(TimeSeriesStateViewer):
    """
    Simple GUI to annotate time series data with non-overlapping state intervals.

    Arguments:
    ----------
    data_axis : matplotlib axis instance
        Axis displaying the time series data.
    state_axis : matplotlib axis instance
        Axis used to display the state annotations.
    key_to_state : dict char : state id (str or int)
        Keyboard keys corresponding to each state.
    state_to_color : dict state id : matplotlib color argument (optional)
        Colors to use for each state in the state annotation plot.
    state_display_order : list of state ids (optional)
        Order of states on the y-axis in state annotation plot.
    default_selection_length : int/float (optional, default 4)
        Default x interval length for a selection.
    default_view_length : int/float (optional, default 60)
        Default x-limit width (i.e. number of time points displayed at any point in time).
    interval_to_state : dict (float start, float stop) : state id (optional, default None)
        Predefined state annotation.
    regions_of_interest: list of (float start, float stop) tuples
        Predefined regions of interest. Press 'h' to see how to quickly navigate these regions.
    selection_callback: function or None (default None)
        Function that is called upon each new selection with selection lower bound
        and selection upper bound as arguments.
    disable_matplotlib_keybindings : bool (default True)
        If True, default matplotlib keybindings are disabled.
        This minimizes conflicts with user defined keybindings.
    verbose : bool (default True)
        If False, warnings are suppressed.

    Example:
    --------
    ```
    import numpy as np
    import matplotlib.pyplot as plt
    from _manual_state_annotation import TimeSeriesAnnotator

    # initialise annotator object
    fig, (data_axis, state_axis) = plt.subplots(2, 1)
    data_axis.plot(np.random.rand(1000))
    keymap = {'a' : 'state A', 'b' : 'state B'}
    annotator = TimeSeriesAnnotator(data_axis, state_axis, keymap)
    plt.show()

    # annotate states by pressing 'a' or 'b'

    # retrieve annotation
    annotation = annotator.interval_to_state
    ```

    Notes:
    ------
    Press 'h' to display the interactive help.
    A summary of all navigation and selection commands can be found there.
    """

    def __init__(self, data_axis, state_axis, key_to_state,
                 disable_matplotlib_keybindings = True,
                 verbose                        = True,
                 *args, **kwargs):

        super(TimeSeriesStateAnnotator, self).__init__(data_axis, state_axis, *args, **kwargs)

        if disable_matplotlib_keybindings:
            if verbose:
                import warnings
                msg =  "Disabling all native matplotlib keyboard shortcuts to minimise conflicts with user-defined keys."
                msg += "\nIf you would like to retain these keybindings, initialise the class with `disable_matplotlib_keybindings` set to False."
                msg += "\nTo supress this warning, initialise the class with `verbose` set to False."
                warnings.warn(msg)
            _disable_matplotlib_keybindings()

        # check that supplied state annotation keys do not conflict with existing key bindings
        self._check_keybindings(key_to_state)

        self.key_to_state = key_to_state

        # initialize state transition markers
        self._initialize_transitions()

        self.figure.canvas.mpl_connect('pick_event', self._on_pick)


    def _on_motion(self, event):
        super(TimeSeriesStateAnnotator, self)._on_motion(event)

        if event.inaxes is self.state_axis:
            if self.picked_transition:
                self._move_transition(event)


    def _on_button_release(self, event):
        super(TimeSeriesStateAnnotator, self)._on_button_release(event)

        if event.inaxes is self.state_axis:
            if self.picked_transition:
                self._update_transition(self.picked_transition)
                self.picked_transition = None


    def _on_pick(self, event):
        if (event.mouseevent.inaxes is self.state_axis) \
           and (event.artist in self.transition_artist_to_interval):
            self.picked_transition = event.artist


    def _on_key_press(self, event):
        # potentially, don.t inherit from parent but copy and extend method instead
        # (if we inherit, we don't have a clean switch/case pattern)
        super(TimeSeriesStateAnnotator, self)._on_key_press(event)

        if event.key in self.key_to_state:
            self._annotate(event)


    def _annotate(self, event):
        if event.key in self.key_to_state:
            self._update_annotation(self.selection_lower_bound,
                                    self.selection_upper_bound,
                                    self.key_to_state[event.key])

        # elif ('alt+' in event.key) and (event.key.replace('alt+', '') in self.key_to_state):
        #     self._update_annotation(self.view_lower_bound,
        #                           self.view_upper_bound,
        #                           self.key_to_state[event.key.replace('alt+', '')])


    def _update_annotation(self, start, stop, state):

        # print(start, stop, state)

        def _is_overlapping(start, stop, start_, stop_):
            # return (start_ <= stop) and (stop_ >= start)
            # return not ((start_ > stop) or (stop_ < start))
            return np.invert(np.bitwise_or((start_ > stop), (stop_ < start)))

        # determine enclosed, enclosing, and overlapping intervals
        if self.interval_to_state:
            intervals = np.array(list(self.interval_to_state.keys()))
            start_, stop_ = intervals.T
            is_affected = _is_overlapping(start, stop, start_, stop_)
            affected_intervals = intervals[is_affected]

        else: # no intervals defined yet
            affected_intervals = []

        # print("Affected intervals:")
        # for interval in affected_intervals:
        #     print(interval)

        # delete old intervals enclosed by new interval
        if len(affected_intervals) > 0:
            start_, stop_ = affected_intervals.T
            is_enclosed = np.logical_and(start_ >= start, stop_ <= stop)
            if np.any(is_enclosed):
                for ii, (start_, stop_) in enumerate(affected_intervals[is_enclosed]):
                    if (start_ >= start) and (stop_ <= stop):
                        # print("New interval is enclosing.")
                        self._delete_interval(start_, stop_)
                affected_intervals = affected_intervals[~is_enclosed]

        # exhaustive list of cases
        total_affected_intervals = len(affected_intervals)

        if total_affected_intervals == 0:
            self._create_interval(start, stop, state)

        elif total_affected_intervals == 1:
            start_, stop_ = affected_intervals[0]
            state_ = self.interval_to_state[(start_, stop_)]

            if (start_ < start) and (stop_ > stop):
                # print("New interval is enclosed.")
                if state != state_:
                    self._delete_interval(start_, stop_)
                    self._create_interval(start_, start, state_)
                    self._create_interval(start, stop, state)
                    self._create_interval(stop, stop_, state_)
                else:
                    # print('The region to update is enclosed by an interval with the same state; i.e. there is nothing that actually needs doing.')
                    pass

            elif ((start_ < start) and (stop_ <= stop)) or (start == stop_): # and (stop_ > start):
                # print("New interval overlaps end of existing interval")
                if state != state_:
                    self._update_interval(start_, stop_, start_, start)
                    self._create_interval(start, stop, state)
                else: # we just need to extend the existing interval
                    self._update_interval(start_, stop_, start_, stop)

            elif ((start_ >= start) and (stop_ > stop)) or (stop == start_): # and (start_ < stop):
                # print("New interval overlaps start of existing interval")
                if state != state_:
                    self._create_interval(start, stop, state)
                    self._update_interval(start_, stop_, stop, stop_)
                else: # we just need to extend the existing interval
                    self._update_interval(start_, stop_, start, stop_)

        elif total_affected_intervals == 2:

            states = [self.interval_to_state[(start_, stop_)] for start_, stop_ in affected_intervals]

            if len(set(states) | {state}) == 1: # case {A, A}; A
                for start_, stop_ in affected_intervals:
                    self._delete_interval(start_, stop_)
                start = np.min(affected_intervals[:, 0], axis=0)
                stop  = np.max(affected_intervals[:, 1], axis=0)
                self._create_interval(start, stop, state)

            else: # case {A, B}; A, {A, A}; B or {A, B}; C

                for start_, stop_ in affected_intervals:
                    state_ = self.interval_to_state[(start_, stop_)]

                    self._delete_interval(start_, stop_)

                    if ((start_ < start) and (stop_ <= stop)) or (start == stop_): # and (stop_ > start):
                        # print("New interval overlaps end of existing interval")
                        if state != state_:
                            self._create_interval(start_, start, state_)
                        else:
                            self._create_interval(start_, stop, state_)

                    elif ((start_ >= start) and (stop_ > stop)) or (stop == start_): # and (start_ < stop):
                        # print("New interval overlaps start of existing interval")
                        if state != state_:
                            self._create_interval(stop, stop_, state_)
                        else:
                            self._create_interval(start, stop_, state_)

                if len(set(states)) < len(set(states) | {state}):
                    # case {A, A}; B or {A, B}; C
                    self._create_interval(start, stop, state)
                else:
                    # case {A, B}; A
                    pass

        else:
            error_msg = "Expected <= 2 intervals to handle but got {}!".format(total_affected_intervals)
            error_msg += "Excluding intervals that are enclosed by the new interval, intervals are:"
            for start_, stop_ in affected_intervals:
                error_msg += "\n{}, {}".format(start_, stop_)
            raise ValueError(error_msg)

        self.figure.canvas.draw_idle()


    def _delete_interval(self, start, stop):
        del self.interval_to_state[(start, stop)]
        self.line_artists[(start, stop)].remove()
        del self.line_artists[(start, stop)]
        self._delete_transition(start, stop)


    def _create_interval(self, start, stop, state):
        self.interval_to_state[(start, stop)] = state
        self.line_artists[(start, stop)], = self.state_axis.plot(
            (start, stop),
            (self.state_to_yvalue[state], self.state_to_yvalue[state]),
            color     = self.state_to_color[state],
            linewidth = STATE_LINE_WIDTH,
        )
        self._create_transition(start, stop)


    def _update_interval(self, old_start, old_stop, new_start, new_stop):
        state = self.interval_to_state[(old_start, old_stop)]
        self._delete_interval(old_start, old_stop)
        self._create_interval(new_start, new_stop, state)


    def _initialize_transitions(self):
        self.picked_transition = None
        self.transition_artist_to_interval = dict()
        self.interval_to_transition_artist = dict()
        for start, stop in self.interval_to_state.keys():
            self._create_transition(start, stop)


    def _create_transition(self, start, stop):
        artist = plt.axvline(stop, picker=10)
        self.transition_artist_to_interval[artist] = (start, stop)
        self.interval_to_transition_artist[(start, stop)] = artist


    def _delete_transition(self, start, stop):
        artist = self.interval_to_transition_artist[(start, stop)]
        artist.remove()
        del self.transition_artist_to_interval[artist]
        del self.interval_to_transition_artist[(start, stop)]


    def _move_transition(self, event):
        xdata, ydata = self.picked_transition.get_data()
        self.picked_transition.set_data([event.xdata, event.xdata], ydata)
        self.figure.canvas.draw_idle()


    def _update_transition(self, transition_artist):
        # determine affected region
        start, stop = self.transition_artist_to_interval[transition_artist]
        xdata, _ = transition_artist.get_data()
        start_, stop_ = sorted([stop, xdata[0]])

        # determine new state
        delta = xdata[0] - stop
        state = self._get_state_at(stop + -EPS * np.sign(delta))

        # update annotation
        self._update_annotation(start_, stop_, state)


    def _get_state_at(self, x):
        interval = self._get_interval_at(x)
        if interval:
            return self.interval_to_state[interval]
        else:
            return None

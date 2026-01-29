ctmc

module user_engagement_timed
    // State encoding (same as DTMC):
    // 0 = browsing (initial)
    // 1 = engaged
    // 2 = disengaged
    // 3 = converted (terminal - success)
    // 4 = abandoned (terminal - failure)

    state : [0..4] init 0;

    // From browsing: rates determine both timing and relative probabilities
    [] state=0 -> 0.50 : (state'=1)    // engage quickly (mean 2s)
                + 0.10 : (state'=2)    // disengage slowly (mean 10s)
                + 0.05 : (state'=4);   // abandon (mean 20s)

    // From engaged: user is actively participating
    [] state=1 -> 0.30 : (state'=3)    // convert (mean ~3.3s)
                + 0.10 : (state'=2)    // lose interest (mean 10s)
                + 0.05 : (state'=4);   // sudden abandon (mean 20s)

    // From disengaged: user has lost interest
    [] state=2 -> 0.15 : (state'=1)    // re-engage (mean ~6.7s)
                + 0.20 : (state'=4);   // abandon (mean 5s)

    // Terminal states: no outgoing transitions
    // (In CTMCs, absorbing states have no transitions)

endmodule

// Labels for property specification
label "browsing" = (state=0);
label "engaged" = (state=1);
label "disengaged" = (state=2);
label "converted" = (state=3);
label "abandoned" = (state=4);
label "done" = (state=3 | state=4);

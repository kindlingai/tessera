//! Steering profiles as edit hooks on writer sites. Sketch.

use crate::core::*;

/// One profile: a direction and strength per steered site.
pub struct Profile {
    pub sites: Vec<(Site, f32)>,
}

/// Loaded profiles; each row picks one by its slot.
pub struct Profiles {
    pub list: Vec<Profile>,
    /// Per site: one direction row per profile.
    pub basis: Vec<(Site, Basis)>,
}

/// Per-row request fields from device row state.
pub struct Request {
    pub slot: RowParam,
    pub scale: RowParam,
}

/// Projects each row's chosen direction out of its writer.
pub fn steer(m: &mut Model, p: &Profiles, req: &Request) {
    for (site, basis) in &p.basis {
        let alpha = m.slot_lookup(&req.slot, &p.alpha_table(site));
        let edit = Edit::Project {
            basis: basis.clone(),
            select: req.slot.clone(),
            coeff: req.scale.mul(&alpha),
        };
        m.hook(*site, Hook::Edit(edit));
    }
}

/// Activation addition instead; affine, so never on a Partial.
pub fn add_vectors(m: &mut Model, p: &Profiles, req: &Request) {
    for (site, basis) in &p.basis {
        let alpha = m.slot_lookup(&req.slot, &p.alpha_table(site));
        let edit = Edit::AddVector {
            basis: basis.clone(),
            select: req.slot.clone(),
            coeff: req.scale.mul(&alpha),
        };
        m.hook(*site, Hook::Edit(edit));
    }
}

/// Caches written after an edited site depend on the profile.
pub fn cache_scope(m: &Model, p: &Profiles) -> Namespaces {
    m.downstream_of(p.basis.iter().map(|(site, _)| *site))
}

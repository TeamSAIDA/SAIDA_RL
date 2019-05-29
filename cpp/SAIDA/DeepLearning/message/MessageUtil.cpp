#include "MessageUtil.h"
#include "common.pb.h"

namespace BWML {
	Message::UInfo *setUInfo(Message::UInfo *uInfo, Unit data)
	{
		uInfo->set_unit_type(data->getType().getName());

		if (!data->exists())
			return uInfo;

		uInfo->set_accelerating(data->isAccelerating());
		uInfo->set_braking(data->isBraking());
		uInfo->set_is_attack_frame(data->isAttackFrame());
		uInfo->set_angle(data->getAngle());
		uInfo->set_attacking(data->isAttacking());
		uInfo->set_cooldown(data->getGroundWeaponCooldown());
		uInfo->set_hp(data->getHitPoints());
		uInfo->set_energy(data->getEnergy());
		uInfo->set_shield(data->getShields());
		uInfo->set_pos_x(data->getPosition().x);
		uInfo->set_pos_y(data->getPosition().y);
		uInfo->set_velocity_x(data->getVelocityX());
		uInfo->set_velocity_y(data->getVelocityY());

		return uInfo;
	}

	Message::UInfo *setUInfo(Message::UInfo *uInfo, MyBot::UnitInfo *data) {
		return setUInfo(uInfo, data->unit());
	}

	Message::TypeInfoMap *setTypeInfo(Message::TypeInfoMap *typeInfoMap, UnitType uType)
	{
		typeInfoMap->set_key(uType.getName());
		Message::TypeInfo *typeInfo = typeInfoMap->mutable_value();

		typeInfo->set_hp_max(uType.maxHitPoints());
		typeInfo->set_shield_max(uType.maxShields());
		typeInfo->set_energy_max(uType.maxEnergy());
		typeInfo->set_armor(uType.armor());
		typeInfo->set_cooldown_max(uType.groundWeapon().damageCooldown());
		typeInfo->set_acceleration(uType.acceleration());
		typeInfo->set_top_speed(uType.topSpeed());
		typeInfo->set_damage_amount(uType.groundWeapon().damageAmount());
		typeInfo->set_damage_factor(uType.groundWeapon().damageFactor());
		typeInfo->set_weapon_range(uType.groundWeapon().maxRange());
		typeInfo->set_sight_range(uType.sightRange());
		typeInfo->set_seek_range(uType.seekRange());

		return typeInfoMap;
	}
}